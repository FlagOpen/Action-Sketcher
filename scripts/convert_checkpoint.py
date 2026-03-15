#!/usr/bin/env python3
"""
Unified Checkpoint Conversion Script

Converts DeepSpeed training checkpoints to HuggingFace format for inference.

Usage:
    python scripts/convert_checkpoint.py \
        --input ckpts/epoch=9-step=13850.ckpt \
        --output ckpts/hf_model

The script handles:
    1. DeepSpeed ZeRO sharded checkpoints (.ckpt directories)
    2. Already-consolidated pytorch_model.bin files
    3. Safetensors format
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from typing import Dict
import json


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load safetensors file."""
    from safetensors.torch import load_file
    return {k: v.detach().cpu() for k, v in load_file(path).items()}


def load_pytorch_bin(path: str) -> Dict[str, torch.Tensor]:
    """Load pytorch .bin file."""
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            return {k: v.detach().cpu() for k, v in loaded["state_dict"].items()}
        if all(isinstance(v, torch.Tensor) for v in loaded.values()):
            return {k: v.detach().cpu() for k, v in loaded.items()}
    raise RuntimeError(f"Unrecognized checkpoint format: {path}")


def load_from_deepspeed_zero(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load weights from DeepSpeed ZeRO checkpoint directory."""
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    ckpt_path = Path(ckpt_dir)
    
    # Check if this is the inner 'checkpoint' directory or parent
    if ckpt_path.name == "checkpoint":
        parent = ckpt_path.parent
        tag = "checkpoint"
    elif (ckpt_path / "checkpoint").exists():
        parent = ckpt_path
        tag = "checkpoint"
    else:
        # Try to find latest tag
        latest_file = ckpt_path / "latest"
        if latest_file.exists():
            tag = latest_file.read_text().strip()
            parent = ckpt_path
        else:
            raise FileNotFoundError(f"Cannot find DeepSpeed checkpoint in {ckpt_dir}")
    
    print(f"[INFO] Loading DeepSpeed ZeRO checkpoint from {parent} with tag='{tag}'")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(parent), tag=tag)
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def load_from_directory(input_dir: str) -> Dict[str, torch.Tensor]:
    """Auto-detect and load checkpoint from directory."""
    input_path = Path(input_dir)
    
    # Check for DeepSpeed ZeRO checkpoint markers
    has_zero_markers = (
        (input_path / "checkpoint").exists() or
        (input_path / "latest").exists() or
        any(input_path.glob("**/zero_pp_rank_*"))
    )
    
    if has_zero_markers:
        print("[INFO] Detected DeepSpeed ZeRO checkpoint")
        return load_from_deepspeed_zero(input_dir)
    
    # Check for safetensors
    st_path = input_path / "model.safetensors"
    if st_path.exists():
        print(f"[INFO] Loading from safetensors: {st_path}")
        return load_safetensors(str(st_path))
    
    # Check for pytorch_model.bin
    bin_path = input_path / "pytorch_model.bin"
    if bin_path.exists():
        print(f"[INFO] Loading from pytorch_model.bin: {bin_path}")
        return load_pytorch_bin(str(bin_path))
    
    # Check for sharded safetensors
    st_index = input_path / "model.safetensors.index.json"
    if st_index.exists():
        print(f"[INFO] Loading sharded safetensors via index")
        with open(st_index) as f:
            index = json.load(f)
        merged = {}
        for shard in sorted(set(index.get("weight_map", {}).values())):
            shard_path = input_path / shard
            if shard_path.exists():
                merged.update(load_safetensors(str(shard_path)))
        return merged
    
    raise FileNotFoundError(
        f"Cannot find valid checkpoint in {input_dir}. "
        f"Expected DeepSpeed ZeRO checkpoint, model.safetensors, or pytorch_model.bin"
    )


def build_model():
    """Build PI0Policy model with default config."""
    from lerobot.common.policies.pi0.configuration_pi0_libero import PI0Config
    from pi0.modeling import PI0Policy
    
    config = PI0Config()
    model = PI0Policy(config)
    return model


def convert_checkpoint(input_path: str, output_path: str, use_safetensors: bool = True):
    """Convert checkpoint to HuggingFace format."""
    
    print(f"\n{'='*60}")
    print(f"Checkpoint Conversion")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Load state dict
    print("[STEP 1/3] Loading checkpoint...")
    state_dict = load_from_directory(input_path)
    print(f"  Loaded {len(state_dict)} parameters")
    
    # Remove 'policy.' prefix if present (from Lightning wrapper)
    if any(k.startswith("policy.") for k in state_dict.keys()):
        print("  Removing 'policy.' prefix from keys")
        state_dict = {k.replace("policy.", "", 1): v for k, v in state_dict.items()}
    
    # Build model and load weights
    print("\n[STEP 2/3] Building model and loading weights...")
    model = build_model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"  Warning: {len(missing)} missing keys (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (showing first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  All weights loaded successfully!")
    
    # Save to HuggingFace format
    print(f"\n[STEP 3/3] Saving to HuggingFace format...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=use_safetensors)
    
    # Save tokenizer
    try:
        model.language_tokenizer.save_pretrained(output_path)
        print("  Tokenizer saved")
    except Exception as e:
        print(f"  Warning: Could not save tokenizer: {e}")
    
    # List output files
    print(f"\n{'='*60}")
    print("Output files:")
    for f in sorted(os.listdir(output_path)):
        size = os.path.getsize(os.path.join(output_path, f))
        size_str = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.2f} MB"
        print(f"  {f} ({size_str})")
    print(f"{'='*60}")
    print("\n✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert DeepSpeed checkpoint
    python scripts/convert_checkpoint.py \
        --input ckpts/epoch=9-step=13850.ckpt \
        --output ckpts/hf_model
    
    # Convert with pytorch .bin format instead of safetensors
    python scripts/convert_checkpoint.py \
        --input ckpts/epoch=9.ckpt \
        --output ckpts/hf_model \
        --no-safetensors
"""
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input checkpoint directory (DeepSpeed .ckpt or HF format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for HuggingFace checkpoint"
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Save as pytorch_model.bin instead of model.safetensors"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    convert_checkpoint(
        input_path=args.input,
        output_path=args.output,
        use_safetensors=not args.no_safetensors
    )


if __name__ == "__main__":
    main()
