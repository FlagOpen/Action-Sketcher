#!/bin/bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456
cd "$(dirname "$0")/../../.."
# Default values (can be overridden by command line arguments)
JSON_PATH="/path/to/your/data/dataset_index.json"
DATA_ROOT="/path/to/your/data"
REASONING_JSON_PATH="/path/to/your/data/compiled_reasoning.json"
NORMALIZATION_PATH="/path/to/your/data/action_stats.json"
VALID_JSON_PATH="/path/to/your/valid_data/test.json"
VALID_REASONING_JSON_PATH="/path/to/your/valid_data/compiled_reasoning.json"

REASONING_ONLY="False"
BALANCE_SAMPLING="False"
USE_VPROMPT="True"
VISUAL_REASONING="True"
ACTION_ONLY="True"
ADD_AUGMENTATION="True"
NO_REFERENCE="False"
MAIN_TASK_PROMPT="False"

EXP_NAME="1203_libero_action_only"
PRETRAINED_MODEL_PATH="./ckpts/stage3_checkpoint"

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
        --action_only)
            ACTION_ONLY="$2"
            shift 2
            ;;
        --add_augmentation)
            ADD_AUGMENTATION="$2"
            shift 2
            ;;
        --no_reference)
            NO_REFERENCE="$2"
            shift 2
            ;;
        --main_task_prompt)
            MAIN_TASK_PROMPT="$2"
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
            echo "Available arguments:"
            echo "  --json_path, --reasoning_json_path, --normalization_path"
            echo "  --valid_json_path, --valid_reasoning_json_path"
            echo "  --reasoning_only, --balance_sampling, --use_vprompt"
            echo "  --visual_reasoning, --action_only, --add_augmentation"
            echo "  --no_reference, --main_task_prompt"
            echo "  --exp_name, --pretrained_model_path"
            exit 1
            ;;
    esac
done

# Change to project root directory

# Run the training script
python -m scripts.train_action_language \
    --json_path "$JSON_PATH" \
    --reasoning_json_path "$REASONING_JSON_PATH" \
    --normalization_path "$NORMALIZATION_PATH" \
    --valid_json_path "$VALID_JSON_PATH" \
    --valid_reasoning_json_path "$VALID_REASONING_JSON_PATH" \
    --reasoning_only "$REASONING_ONLY" \
    --balance_sampling "$BALANCE_SAMPLING" \
    --use_vprompt "$USE_VPROMPT" \
    --visual_reasoning "$VISUAL_REASONING" \
    --action_only "$ACTION_ONLY" \
    --add_augmentation "$ADD_AUGMENTATION" \
    --no_reference "$NO_REFERENCE" \
    --main_task_prompt "$MAIN_TASK_PROMPT" \
    --data_root "$DATA_ROOT" \
    --exp_name "$EXP_NAME" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --train_expert_only "True"