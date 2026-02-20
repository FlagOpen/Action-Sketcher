"""
PI0 LIBERO Evaluation Script with Visual Reasoning

This script evaluates PI0 model on LIBERO benchmark tasks with full
visual reasoning capabilities including subtask decomposition and 
visual prompt annotations.

Usage:
    python run_libero_example.py --checkpoint ./checkpoint --task_id 0
    python run_libero_example.py --checkpoint ./checkpoint --all_tasks
"""

import argparse
import json
import re
import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gym
import imageio
import numpy as np
import torch

# LIBERO environment (install separately)
from cleandiffuser.env import libero  # noqa: F401

import inference
from draw_vp import visualize_generated_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="PI0 LIBERO Evaluation")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint",
                        help="Path to model checkpoint directory")
    parser.add_argument("--task_id", type=int, default=0,
                        help="LIBERO task ID (0-9)")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Run evaluation on all 10 tasks")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes per task")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--env_type", type=str, default="libero-10-v0",
                        choices=["libero-spatial-v0", "libero-goal-v0", "libero-object-v0","libero-10-v0"],
                        help="LIBERO environment type")
    parser.add_argument("--output_dir", type=str, default="./outputs/long2",
                        help="Output directory for videos and results")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save video of episodes")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Random seed")
    return parser.parse_args()


# ==================== Helper Functions ====================

def extract_two_answers(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract first two <answer>...</answer> contents from text.
    
    Returns:
        (first_answer, second_answer): Subtask and visual prompts, or None if not found
    """
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    first = matches[0].strip() if len(matches) > 0 else None
    second = matches[1].strip() if len(matches) > 1 else None
    return first, second


def format_visual_prompts(visual_reasoning_text: str) -> str:
    """Format visual reasoning text."""
    return visual_reasoning_text.strip()


def create_prompt(main_task: str, subtask: str, visual_reasoning_text: Optional[str] = None) -> str:
    """Construct prompt from core components.
    
    Args:
        main_task: The high-level task instruction
        subtask: Current subtask to execute
        visual_reasoning_text: Optional visual prompts JSON string
        
    Returns:
        Formatted prompt string for the model
    """
    prompt_parts = [
        f"The high-level instruction is {main_task}. Now I need to do the subtask {subtask}."
    ]
    if visual_reasoning_text:
        formatted_prompts = format_visual_prompts(visual_reasoning_text)
        prompt_parts.append(
            f"To guide me in achieving this subtask, I will use the following visual prompts {formatted_prompts}."
        )
    
    prompt_parts.append(
        "Please observe whether we have completed this subtask. "
        "If it has been completed, think about the next subtask to achieve the high-level instruction. "
        "If not, continue the action"
    )
    return "\n".join(prompt_parts)


def load_action_stats(checkpoint_path: str) -> dict:
    """Load action normalization statistics."""
    stats_path = Path(checkpoint_path) / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found in {checkpoint_path}")
    with open(stats_path) as f:
        return json.load(f)


def create_denormalizer(stats: dict, action_dim: int = 7):
    """Create action denormalization function.
    
    Args:
        stats: Dictionary with min/max action bounds
        action_dim: Action dimensionality (default 7 for LIBERO)
        
    Returns:
        Function that denormalizes actions from [-1, 1] to original scale
    """
    action_min = torch.as_tensor(stats["min"][:action_dim], dtype=torch.float32)
    action_max = torch.as_tensor(stats["max"][:action_dim], dtype=torch.float32)
    gripper_idx = action_dim - 1
    
    def denormalize(norm_action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] -> original scale (gripper unchanged)."""
        raw = norm_action.clone()
        for i in range(action_dim):
            if i == gripper_idx:
                continue
            span = (action_max[i] - action_min[i]).clamp_min(1e-8)
            raw[..., i] = (norm_action[..., i] + 1.0) / 2.0 * span + action_min[i]
        return raw
    
    return denormalize


# ==================== Main Evaluation ====================

def evaluate_task(policy, env, task_name: str, denormalize_fn, device: str,
                  num_episodes: int, max_steps: int, output_dir: Path, 
                  task_id: int, save_video: bool = True) -> dict:
    """Evaluate policy on a single LIBERO task.
    
    Args:
        policy: Loaded PI0 policy
        env: LIBERO gym environment
        task_name: Task description string
        denormalize_fn: Action denormalization function
        device: Torch device string
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        output_dir: Directory for saving outputs
        task_id: Task ID for naming
        save_video: Whether to save episode videos
        
    Returns:
        Dictionary with task results
    """
    action_dim = 7
    task_dir = output_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"task_name": task_name, "episodes": []}
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, done = env.reset(), False
        
        # Wait for environment to stabilize
        for _ in range(20):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
        
        # Initialize session state
        session = {
            "main_task": task_name,
            "current_prompt": task_name,
            "ref_0_rgb": None,
        }
        inference.reset_session(session["current_prompt"])
        
        frames = []
        step_count = 0
        
        while not done and step_count < max_steps:
            # Get current robot state (pos + axis-angle + gripper)
            try:
                import robosuite.utils.transform_utils as T
                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ], dtype=np.float32)
            except ImportError:
                # Fallback without robosuite transform utils
                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"][:3],
                    obs["robot0_gripper_qpos"],
                ], dtype=np.float32)
            
            # Get images (convert BGR to RGB)
            base_rgb = obs["agentview_image"][:, :, ::-1].copy()
            wrist_rgb = obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            
            # Initialize reference image if not set
            if session["ref_0_rgb"] is None:
                session["ref_0_rgb"] = base_rgb.copy()
            
            # Prepare inputs
            image_obs = {
                "base_0_rgb": base_rgb,
                "left_wrist_0_rgb": wrist_rgb,
                "ref_0_rgb": session["ref_0_rgb"],
            }
            state_tensor = torch.from_numpy(state).to(device, dtype=torch.bfloat16)[None, None, :]
            
            # Run inference
            result = inference.inference(
                prompt=session["current_prompt"],
                image_obs=image_obs,
                state=state_tensor,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=768,
                device=device,
                reset=True,
            )
            
            if result["is_action"]:
                # === ACTION BRANCH ===
                actions = result["action"].to(dtype=torch.float32).cpu()
                actions = actions[:, :action_dim]
                actions_denorm = denormalize_fn(actions).numpy()
                
                # Execute actions
                for action in actions_denorm:
                    if done:
                        break
                    obs, _, done, _ = env.step(action[:7])
                    frames.append(obs["agentview_image"].transpose(1, 2, 0))
                    step_count += 1
                    
            else:
                # === REASONING BRANCH ===
                generated_text = result["text"]
                print(f"    Reasoning: {generated_text[:100]}...")
                
                # Extract subtask and visual prompts
                subtask, visual_prompts = extract_two_answers(generated_text)
                
                if subtask and visual_prompts:
                    print(f"    Subtask: {subtask}")
                    print(f"    Visual prompts: {visual_prompts[:80]}...")
                    
                    # Generate annotated image
                    image_tensor = torch.from_numpy(base_rgb)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"task{task_id}_ep{episode}_{timestamp}.png"
                    
                    annotated_img = visualize_generated_prompts(
                        image_tensor=image_tensor,
                        generated_text=generated_text,
                        output_filename=filename
                    )
                    
                    # Create new prompt with subtask and visual reasoning
                    new_prompt = create_prompt(
                        main_task=session["main_task"],
                        subtask=subtask,
                        visual_reasoning_text=visual_prompts
                    )
                    session["current_prompt"] = new_prompt
                    
                    # Update reference image with annotated version
                    if annotated_img is not None:
                        session["ref_0_rgb"] = np.transpose(annotated_img, (2, 0, 1))
                    
                    # Reset inference session with new prompt
                    inference.reset_session(session["current_prompt"])
                else:
                    print(f"    Warning: Could not extract answers from reasoning")
                    break
        
        # Record result
        success = bool(done) and step_count < max_steps
        results["episodes"].append(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"    -> {status} (steps: {step_count})")
        
        # Save video
        if save_video and frames:
            video_path = task_dir / f"ep{episode}_{status}.mp4"
            writer = imageio.get_writer(str(video_path), fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"    Video: {video_path}")
    
    return results


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading PI0 model from: {args.checkpoint}")
    policy = inference.load_policy(args.checkpoint)
    
    # Setup device and dtype
    if torch.cuda.is_available():
        policy = policy.to(dtype=torch.bfloat16)
        device = "cuda"
        print("Using CUDA with bfloat16")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Load normalization stats
    stats = load_action_stats(args.checkpoint)
    denormalize_fn = create_denormalizer(stats)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine tasks to evaluate
    if args.all_tasks:
        task_ids = range(10)
    else:
        task_ids = [args.task_id]
    
    all_results = {}
    
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        print(f"{'='*60}")
        
        # Create environment
        env = gym.make(
            args.env_type,
            task_id=task_id,
            image_size=224,
            camera_names=["agentview", "robot0_eye_in_hand"],
            seed=args.seed,
        )
        
        task_name = env.task_description
        print(f"Task: {task_name}")
        
        # Evaluate
        results = evaluate_task(
            policy=policy,
            env=env,
            task_name=task_name,
            denormalize_fn=denormalize_fn,
            device=device,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            output_dir=output_dir,
            task_id=task_id,
            save_video=args.save_video,
        )
        
        all_results[task_id] = results
        env.close()
        
        # Print task summary
        successes = sum(results["episodes"])
        total = len(results["episodes"])
        print(f"Task {task_id} success rate: {successes}/{total} ({100*successes/total:.1f}%)")
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_id, results in all_results.items():
        successes = sum(results["episodes"])
        total = len(results["episodes"])
        print(f"Task {task_id} ({results['task_name'][:40]}...): {successes}/{total} ({100*successes/total:.1f}%)")
    
    total_successes = sum(sum(r["episodes"]) for r in all_results.values())
    total_episodes = sum(len(r["episodes"]) for r in all_results.values())
    print(f"\nOverall: {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)")


if __name__ == "__main__":
    main()
