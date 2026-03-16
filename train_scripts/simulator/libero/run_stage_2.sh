#!/bin/bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456
cd "$(dirname "$0")/../../.."
# Default values (can be overridden by command line arguments)
JSON_PATH="//share/project/xxq/data/liberofinal/dataset_index.json"
DATA_ROOT="/share/project/xxq/data/liberofinal"
REASONING_JSON_PATH="/share/project/xxq/data/liberofinal/compiled_reasoning.json"
NORMALIZATION_PATH="/share/project/xxq/data/liberofinal/action_stats.json"
VALID_JSON_PATH="/path/to/your/valid_data/test.json"
VALID_REASONING_JSON_PATH="/path/to/your/valid_data/compiled_reasoning.json"

REASONING_ONLY="True"
BALANCE_SAMPLING="False"
USE_VPROMPT="True"
VISUAL_REASONING="True"

EXP_NAME="1214_libero"
PRETRAINED_MODEL_PATH="/share/project/xxq/pytorch_pi/ckpts/stage1_robobrain_hf_libero"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json_path)
            JSON_PATH="$2"
            shift 2
            ;;
        --reasoning_json_path)
            REASONING_JSON_PATH="$2"
            shift 2
            ;;
        --normalization_path)
            NORMALIZATION_PATH="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --valid_json_path)
            VALID_JSON_PATH="$2"
            shift 2
            ;;
        --valid_reasoning_json_path)
            VALID_REASONING_JSON_PATH="$2"
            shift 2
            ;;
        --reasoning_only)
            REASONING_ONLY="$2"
            shift 2
            ;;
        --balance_sampling)
            BALANCE_SAMPLING="$2"
            shift 2
            ;;
        --use_vprompt)
            USE_VPROMPT="$2"
            shift 2
            ;;
        --visual_reasoning)
            VISUAL_REASONING="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --pretrained_model_path)
            PRETRAINED_MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run the pi0 language training script
python -m scripts.train_language \
    --json_path "$JSON_PATH" \
    --reasoning_json_path "$REASONING_JSON_PATH" \
    --normalization_path "$NORMALIZATION_PATH" \
    --valid_json_path "$VALID_JSON_PATH" \
    --valid_reasoning_json_path "$VALID_REASONING_JSON_PATH" \
    --reasoning_only "$REASONING_ONLY" \
    --balance_sampling "$BALANCE_SAMPLING" \
    --use_vprompt "$USE_VPROMPT" \
    --visual_reasoning "$VISUAL_REASONING" \
    --data_root "$DATA_ROOT" \
    --exp_name "$EXP_NAME" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH"