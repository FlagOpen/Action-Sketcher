"""
PI0 Inference Module for LIBERO

This module provides a simple interface for running inference with the PI0 model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

# Import PI0 model and config
from pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
import lerobot.common.policies.pi0.configuration_pi0_libero  # Register pi0_libero config

_POLICY: Optional[PI0Policy] = None
_DEVICE: Optional[torch.device] = None
_SESSION_PROMPT: Optional[str] = None


def _select_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_policy(checkpoint_path: str, device: Optional[str] = None) -> PI0Policy:
    """Load PI0 policy from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory
        device: Device to load model on (cuda/cpu)
    
    Returns:
        Loaded PI0Policy model
    """
    global _POLICY, _DEVICE
    
    if _POLICY is not None:
        return _POLICY
    
    _DEVICE = _select_device(device)
    
    policy = PI0Policy.from_pretrained(checkpoint_path).eval()
    policy.to(_DEVICE)
    
    _POLICY = policy
    return policy


TensorLikeImage = Union[torch.Tensor, np.ndarray, Image.Image]
TensorLikeState = Union[torch.Tensor, np.ndarray, list, tuple]


def _to_3chw_tensor(img: TensorLikeImage) -> torch.Tensor:
    """Convert image to 3xHxW torch.Tensor (values in [0,255], dtype=float32)."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        arr = np.array(img)
        t = torch.from_numpy(arr)
        t = t.permute(2, 0, 1)
    elif isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif t.ndim == 3 and t.shape[0] == 3:
            pass
        else:
            raise ValueError("Unsupported numpy image shape. Expect HxWx3 or 3xHxW.")
    elif isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif t.ndim == 3 and t.shape[0] == 3:
            pass
        else:
            raise ValueError("Unsupported tensor image shape.")
    else:
        raise TypeError("Unsupported image type.")

    if not torch.is_floating_point(t):
        t = t.to(torch.float32)
    else:
        t = t.float()
    return t


def _prepare_observation(
    image_obs: Dict[str, TensorLikeImage],
    policy: PI0Policy,
    prompt_text: str,
    state_in: Optional[TensorLikeState] = None,
) -> Dict[str, Any]:
    """Prepare observation dict for policy."""
    device = next(policy.parameters()).device
    
    try:
        param_dtype = next(policy.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32
    
    image_tensors: Dict[str, torch.Tensor] = {}
    for k, v in image_obs.items():
        t = _to_3chw_tensor(v)
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"Image for key {k} must be 3xHxW after conversion.")
        image_tensors[k] = t.unsqueeze(0).to(device)

    observation: Dict[str, Any] = {
        "image": image_tensors,
        "state": state_in,
        "prompt": [prompt_text],
    }
    return observation


def reset_session(prompt: Optional[str] = None) -> None:
    """Reset session memory. Optionally set a new starting prompt."""
    global _SESSION_PROMPT
    _SESSION_PROMPT = prompt


@torch.no_grad()
def inference(
    prompt: str,
    image_obs: Dict[str, TensorLikeImage],
    state: Optional[TensorLikeState] = None,
    checkpoint_path: Optional[str] = None,
    *,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_k: int = 0,
    max_new_tokens: int = 768,
    device: Optional[str] = None,
    reset: bool = False,
) -> Dict[str, Any]:
    """Run single-step inference (batch=1).

    Args:
        prompt: Text prompt for the model
        image_obs: Dict of image observations (base_0_rgb, left_wrist_0_rgb, ref_0_rgb)
        state: Robot state tensor (S,) or (1,S)
        checkpoint_path: Path to model checkpoint (required on first call)
        do_sample: Whether to sample or use greedy decoding
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum new tokens to generate
        device: Device to run on
        reset: Whether to reset session prompt

    Returns:
        Dict containing:
        - is_action: bool - whether model output actions or reasoning
        - text: the prompt or generated text
        - action: tensor of actions (or zeros if reasoning)
    """
    global _SESSION_PROMPT

    if _POLICY is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required on first call")
        load_policy(checkpoint_path, device)
    
    policy = _POLICY

    if reset or _SESSION_PROMPT is None:
        if not isinstance(prompt, str) or len(prompt) == 0:
            raise ValueError("First inference requires non-empty prompt.")
        _SESSION_PROMPT = prompt

    used_prompt = _SESSION_PROMPT

    observation = _prepare_observation(
        image_obs=image_obs,
        policy=policy,
        prompt_text=used_prompt,
        state_in=state,
    )

    out = policy.action_or_reasoning(
        observation=observation,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        return_text=True,
    )

    is_action = bool(out.get("is_action", False))

    if is_action:
        actions = out["actions"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions.detach().to("cpu")
        if actions.dim() == 3 and actions.size(0) == 1:
            actions = actions.squeeze(0)

        return {
            "is_action": True,
            "text": used_prompt,
            "action": actions,
        }

    texts = out.get("texts", [""])
    gen_text = texts[0] if isinstance(texts, (list, tuple)) and len(texts) > 0 else ""
    _SESSION_PROMPT = gen_text

    T = int(policy.config.n_action_steps)
    A = int(policy.config.max_action_dim)
    zero_action = torch.zeros((T, A), dtype=torch.float32)

    return {
        "is_action": False,
        "text": gen_text,
        "action": zero_action,
    }
