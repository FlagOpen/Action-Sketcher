import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Dict, Any, Optional
import os
import re
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import random

def _find_image_any_ext(stem_dir: Path, stem_name: str) -> str:
    """Return full path with the extension that actually exists."""
    for ext in ('.jpg', '.png'):
        p = stem_dir / f"{stem_name}{ext}"
        if p.exists():
            return str(p)
    # fallback – return jpg even if missing (caller will catch it)
    return str(stem_dir / f"{stem_name}.jpg")


def format_visual_prompts(prompt_dict: Dict) -> str:
    """Formats the visual prompt dictionary into a clean string."""
    if not isinstance(prompt_dict, dict):
        return "{}"
    parts = []
    for key, value in prompt_dict.items():
        if value:  # Only include prompts that are not empty
            parts.append(f"'{key}': {value}")
    return f"{{{', '.join(parts)}}}"

class RoboticsDataset(Dataset):
    """机器人数据集加载器（带视觉提示与阶段化推理提示）。

    功能：
    - 从 episodes JSON 中读取段（segment），包括 reasoning / action 两类；
    - 为首个 reasoning 段生成主任务提示，其余段根据上一个关键帧生成上下文提示；
    - 可选视觉提示（vprompt）：用标注后的参考图替换普通参考图路径；
    - 输出与 `PI0Policy.forward` 兼容的字典：`image`/`state`/`action`/`prompt`；

    关键返回：
    - image：包含 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`、`ref_0_rgb` 四键；
    - state：形状 [1, 8]，无归一化；
    - action：形状 [10, 7]，前6维使用min/max归一化到[-1, 1]，gripper维度(last dim)保持原始值；
    - prompt：List[str] = [前缀文本, 后缀文本]，后缀以 <BEGIN_OF_REASONING> 或 <BEGIN_OF_ACTION> 开头；
    """
    def __init__(
        self,
        json_path: str,
        reasoning_json_path: str,
        normalization_path: Optional[str] = None,
        data_root: str = "",
        image_transform: Optional[transforms.Compose] = None,
        use_vprompt: bool = False,
        visual_reasoning: bool = False,
        balance_sampling: bool = False,
        reasoning_only: bool = False,
        action_only: bool = False,
        main_task_prompt: bool = False,
        no_reference: bool = False,
        add_augmentation: bool = False,
        augment_ratio: float = 0.1,
    ):
        with open(json_path, 'r') as f:
            self.data = json.load(f).get('episodes', {})
        
        self.data_root = Path(data_root) if data_root else None

        self.transform = image_transform if image_transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).clamp(0, 255).to(torch.uint8))
        ])
        
        self.main_task_prompt = main_task_prompt
        self.use_vprompt = use_vprompt
        self.visual_reasoning = visual_reasoning
        self.balance_sampling = balance_sampling
        self.reasoning_only = reasoning_only
        self.action_only = action_only
        self.no_reference = no_reference
        self.add_augmentation = add_augmentation
        self.augment_ratio = augment_ratio
        
        # MODIFIED: 更新占位符形状为 (8,) 和 (10, 7)
        self.qpos_placeholder = np.zeros((8,), dtype=np.float32)
        self.action_placeholder = np.zeros((10, 7), dtype=np.float32)

        # MODIFIED: 仅加载动作归一化参数（min/max）
        self.action_min, self.action_max = None, None
        if normalization_path and os.path.exists(normalization_path):
            self._load_normalization_params(normalization_path)
        
        self.reasoning_data = {}
        if reasoning_json_path and os.path.exists(reasoning_json_path):
            with open(reasoning_json_path, 'r') as f:
                self.reasoning_data = json.load(f)

        self.samples = []
        self.reasoning_indices = []
        self.action_indices = []
        self._process_episodes()

        if self.balance_sampling:
            self._setup_balanced_sampling()

    def _load_normalization_params(self, normalization_path: str):
        """MODIFIED: 从JSON加载动作min/max统计信息"""
        with open(normalization_path, 'r') as f:
            norm_data = json.load(f)
            # 加载动作统计信息
            self.action_min = np.array(norm_data["min"])
            self.action_max = np.array(norm_data["max"])
            
            # 验证维度匹配
            assert self.action_min.shape == (7,), f"Expected action min shape (7,), got {self.action_min.shape}"
            assert self.action_max.shape == (7,), f"Expected action max shape (7,), got {self.action_max.shape}"

    def _process_episodes(self):
        total_episodes, global_idx = len(self.data), 0
        
        for episode_id, episode_data in self.data.items():
            segments = episode_data.get('segments', [])
            if not segments: continue
            
            episode_dir = Path(segments[0].get('state', [""])[0]).parents[1]
            if self.data_root:
                episode_dir = self.data_root / episode_dir
            episode_logs = self.reasoning_data.get(f"episode_{episode_id}", {})

            # --- START: MODIFIED main_task LOGIC ---
            main_task = None
            instruction_json_path = episode_dir / "instructions.json"

            # 1. Try to load from instruction.json first
            try:
                if instruction_json_path.exists():
                    with open(instruction_json_path, 'r') as f:
                        instruction_data = json.load(f)
                    instructions_list = instruction_data.get("instructions", [])
                    if instructions_list:
                        main_task = random.choice(instructions_list)
            except Exception as e:
                print(f"Warning: Error reading {instruction_json_path}: {e}. Falling back.")
                main_task = None # Ensure fallback
            
            try:
                annotation_path = next(episode_dir.glob("*_resized.json"))
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)
                if main_task is None:  # Fallback: main_task not set by instruction.json
                    main_task = annotations.get("main_task", "Stack the blocks in this order: Red, Green, Blue")
                annotation_lookup = {
                    int(data['frame_id']): {
                        'subtask': data.get('subtask', ''),
                        'visual_prompt': data.get('resized_visual_prompts', {})
                    }
                    for key, data in annotations.items() if key.startswith("key_frame")
                }
            except (StopIteration, FileNotFoundError):
                if main_task is None: # Fallback: neither file worked
                    main_task = "pick up the orange and place into the basket"
                annotation_lookup = {}

            full_log_lookup = {
                int(k.split('_')[1]): {"reasoning": v.get("reasoning", ""), "visual_reasoning": v.get("visual_reasoning", "")}
                for k, v in episode_logs.items()
            }
            reasoning_indices = sorted(list(annotation_lookup.keys()))
            
            for segment in segments:
                if segment.get('type') == 'spatial' : continue
                if self.reasoning_only and segment.get('type') == 'action' : continue
                if self.action_only and segment.get('type') != 'action': continue
                if segment.get('type') == 'action' and not segment.get('action_chunk'): continue
                
                prompt, output = "", ""
                
                if segment['type'] == 'reasoning' or segment['type'] == 'temporal' :
                    current_frame = segment['frame_index']
                    current_log = full_log_lookup.get(current_frame, {})
                    output = f"{current_log.get('reasoning', '')}\n{current_log.get('visual_reasoning', '')}".strip() if self.visual_reasoning else current_log.get('reasoning', '')
                    
                    if '{{}}' in output:
                        output = output.replace('{{}}', main_task)

                    try:
                        current_reasoning_idx = reasoning_indices.index(current_frame)
                    except ValueError: continue

                    # --- MODIFICATION: Special prompt for the first reasoning segment ---
                    if current_reasoning_idx == 0:
                        prompt = main_task
                    else:
                        # Logic for subsequent reasoning segments
                        context_frame_idx = reasoning_indices[current_reasoning_idx - 1]
                        context_data = annotation_lookup.get(context_frame_idx, {})
                        subtask = context_data.get('subtask', "")
                        visual_prompts = context_data.get('visual_prompt', {})
                        
                        prompt_parts = [f"The high-level instruction is '{main_task}'. Now I need to do the subtask '{subtask}'."]
                        if self.visual_reasoning and visual_prompts:
                            formatted_prompts = format_visual_prompts(visual_prompts)
                            prompt_parts.append(f"To guide me in achieving this subtask, I will use the following visual prompts '{formatted_prompts}'.")
                        prompt_parts.append("Please observe whether we have completed this subtask. If it has been completed, think about the next subtask to achieve the high-level instruction. if not, continue the action")
                        prompt = "\n".join(prompt_parts)

                elif segment['type'] == 'action':
                    output = "<action_token>"
                    # MODIFIED: Conditional prompt based on annotation availability
                    if self.main_task_prompt:
                        prompt = main_task
                    else:
                        if annotation_lookup:
                            context_frame_idx = segment['reference_frame_index']
                            context_data = annotation_lookup.get(context_frame_idx, {})
                            subtask = context_data.get('subtask', "")
                            visual_prompts = context_data.get('visual_prompt', {})

                            prompt_parts = [f"The high-level instruction is '{main_task}'. Now I need to do the subtask '{subtask}'."]
                            if self.visual_reasoning and visual_prompts:
                                formatted_prompts = format_visual_prompts(visual_prompts)
                                prompt_parts.append(f"To guide me in achieving this subtask, I will use the following visual prompts '{formatted_prompts}'.")
                            prompt_parts.append("Please observe whether we have completed this subtask. If it has been completed, think about the next subtask to achieve the high-level instruction. if not, continue the action")
                            prompt = "\n".join(prompt_parts)
                        else:
                            prompt = main_task

                ref_frame_idx = segment.get('reference_frame_index')
                ref_images = []
                if ref_frame_idx != "":
                    if self.use_vprompt and not (segment.get('type') == 'temporal' and segment.get('frame_index') == ref_frame_idx):
                        vprompt_stem = episode_dir / "visual_prompts" / f"annotated_frame_{ref_frame_idx:06d}"
                        vprompt_path = _find_image_any_ext(vprompt_stem.parent, vprompt_stem.stem)
                        ref_images = [str(vprompt_path)]
                        if self.add_augmentation and segment.get('type') == 'action':
                            aug_stem = episode_dir / "visual_prompts" / f"annotated_frame_{ref_frame_idx:06d}_augmented"
                            aug_path = _find_image_any_ext(aug_stem.parent, aug_stem.stem)
                            if Path(aug_path).exists():
                                ref_images.append(aug_path)
                    else:
                        ref_segment = next((s for s in segments if s['frame_index'] == ref_frame_idx), None)
                        if ref_segment: ref_images = ref_segment.get('image', [])
                
                sample = {
                    'episode_id': episode_id, 'type': segment.get('type', ''), 'frame_index': segment.get('frame_index', ''),
                    'images': segment.get('image', []), 'reference_images': ref_images,
                    'image_paths': segment.get('image', []), 'ref_image_paths': ref_images,
                    'state_path': segment.get('state', [])[0] if segment.get('state') else None,
                    'action_chunk_path': segment.get('action_chunk', ''),
                    'prompt': prompt, 'output': output,
                    'reference_frame_index': segment.get('reference_frame_index','')
                }
                self.samples.append(sample)
                
                if segment.get('type') == 'reasoning' or segment.get('type') == 'temporal': self.reasoning_indices.append(global_idx)
                elif segment.get('type') == 'action': self.action_indices.append(global_idx)
                global_idx += 1

        print(f"Loading dataset: {total_episodes} episodes, {len(self.samples)} valid segments loaded")
        print(f"  - Reasoning samples: {len(self.reasoning_indices)}")
        print(f"  - Action samples: {len(self.action_indices)}")

    def _setup_balanced_sampling(self):
        print("Setting up balanced sampling for reasoning vs. action...")
        num_reasoning, num_action = len(self.reasoning_indices), len(self.action_indices)
        if num_reasoning == 0 or num_action == 0:
            print("Warning: One sample type has zero instances. Balanced sampling disabled.")
            self.balance_sampling = False; return
        weights = torch.zeros(len(self.samples))
        if num_action > num_reasoning:
            weights[self.action_indices], weights[self.reasoning_indices] = 1.0, num_action / num_reasoning
        else:
            weights[self.reasoning_indices], weights[self.action_indices] = 1.0, num_reasoning / num_action
        self.sampler = WeightedRandomSampler(weights, num_samples=len(self.samples), replacement=True)
        print("✅ Balanced sampler created.")

    def get_sampler(self):
        if self.balance_sampling and hasattr(self, 'sampler'): return self.sampler
        return None
        
    def __len__(self) -> int:
        if self.balance_sampling:
            # Length = 2x the smaller set (ensures both sampled equally)
            return 2 * min(len(self.action_indices), len(self.reasoning_indices))
        elif self.reasoning_only:
            return len(self.reasoning_indices)
        elif self.action_only:
            return len(self.action_indices)
        else:
            return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map idx based on sampling mode
        if self.balance_sampling:
            # Alternate: even idx -> action, odd idx -> reasoning
            if idx % 2 == 0:
                real_idx = self.action_indices[(idx // 2) % len(self.action_indices)]
            else:
                real_idx = self.reasoning_indices[(idx // 2) % len(self.reasoning_indices)]
        elif self.reasoning_only:
            real_idx = self.reasoning_indices[idx % len(self.reasoning_indices)]
        elif self.action_only:
            real_idx = self.action_indices[idx % len(self.action_indices)]
        else:
            real_idx = idx
        
        sample = self.samples[real_idx]
        images = {}
        cam_tags = ['base', 'left_wrist', 'right_wrist']
        for tag, path in zip(cam_tags, sample['images']):
            if self.data_root:
                path = str(self.data_root / path)
            try:
                img = Image.open(path).convert('RGB')
                images[f"{tag}_0_rgb"] = self.transform(img)
            except Exception:
                images[f"{tag}_0_rgb"] = torch.zeros(3, 224, 224, dtype=torch.uint8)
        if sample['reference_images'] and not self.no_reference:
            try:
                use_aug = self.add_augmentation and \
                      len(sample['reference_images']) == 2 and \
                      random.random() < self.augment_ratio
                idx = 1 if use_aug else 0
                ref_path = sample['reference_images'][idx]
                if self.data_root:
                    ref_path = str(self.data_root / ref_path)
                img = Image.open(ref_path).convert('RGB')
                images["ref_0_rgb"] = self.transform(img)
            except Exception:
                images["ref_0_rgb"] = torch.zeros(3, 224, 224, dtype=torch.uint8)
        else:
             images["ref_0_rgb"] = torch.zeros(3, 224, 224, dtype=torch.uint8)

        # MODIFIED: 加载动作（10, 7），应用min/max归一化（gripper除外）
        action_chunk_path = sample['action_chunk_path']
        if self.data_root and action_chunk_path:
            action_chunk_path = str(self.data_root / action_chunk_path)
        if action_chunk_path and os.path.exists(action_chunk_path):
            action_chunk = np.load(action_chunk_path)  # Shape: (10, 7)
            
            # MODIFIED: 仅对前6个维度进行归一化，gripper维度(last dim)保持不变
            if self.action_min is not None and self.action_max is not None:
                # 计算除最后一维外的安全范围
                range_safe = np.maximum(self.action_max[:-1] - self.action_min[:-1], 1e-8)
                
                # 对前6个维度应用min/max归一化到[-1, 1]
                action_chunk[:, :-1] = 2.0 * (action_chunk[:, :-1] - self.action_min[:-1]) / range_safe - 1.0
                action_chunk[:, :-1] = np.clip(action_chunk[:, :-1], -1.0, 1.0)
                
                # 最后一维（gripper）保持原始值，不做归一化
            action_tensor = torch.from_numpy(action_chunk).float()
        else:
            action_tensor = torch.from_numpy(self.action_placeholder).float()

        # MODIFIED: 加载状态（8,），无归一化
        state_path = sample['state_path']
        if self.data_root and state_path:
            state_path = str(self.data_root / state_path)
        if state_path and os.path.exists(state_path):
            state = np.load(state_path)  # Shape: (8,)
            state_tensor = torch.from_numpy(state).float()
        else:
            state_tensor = torch.from_numpy(self.qpos_placeholder).float()

        if sample['output'].startswith("<action_token>"):
            output = '<BEGIN_OF_ACTION>'
        else:
            output = "<BEGIN_OF_REASONING>" + sample['output']
        
        return {
            'image': images, 
            'action': action_tensor, 
            'state': state_tensor[None,:],  # Shape: [1, 8]
            'prompt': [sample['prompt'] , output],
            'image_paths': sample['image_paths'],
            'ref_image_paths': sample['ref_image_paths'],
            'episode_id': sample['episode_id'],
            'frame_index': sample['frame_index']
        }

def robotics_collate_fn(batch):
    prompts = [item.pop('prompt') for item in batch]
    image_paths = [item.pop('image_paths') for item in batch]
    ref_image_paths = [item.pop('ref_image_paths') for item in batch]
    episode_ids = [item.pop('episode_id') for item in batch]
    frame_indices = [item.pop('frame_index') for item in batch]
    collated = default_collate(batch)
    collated['prompt'] = prompts
    collated['image_paths'] = image_paths
    collated['ref_image_paths'] = ref_image_paths
    collated['episode_id'] = episode_ids
    collated['frame_index'] = frame_indices
    return collated