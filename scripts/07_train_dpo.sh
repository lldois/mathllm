#!/bin/bash
# =============================================================================
# DPO Training (Direct Preference Optimization)
# Trains on preference pairs: correct solutions (chosen) vs incorrect (rejected)
# Usage: bash scripts/07_train_dpo.sh <model_path> <dpo_data_path>
# =============================================================================
set -e

MODEL_PATH="${1:?Usage: $0 <model_path> [dpo_data_path]}"
DATA_PATH="${2:-data_processing/processed/dpo_pairs.jsonl}"
OUTPUT_DIR="${3:-outputs/dpo/dpo_experiment}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mathmodel-reasoning}"

cd "$(dirname "$0")/.."

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: DPO data not found at $DATA_PATH"
    echo "Run: bash scripts/06_generate_dpo_data.sh <model_path>"
    exit 1
fi

PAIRS=$(wc -l < "$DATA_PATH")
echo "============================================"
echo "DPO Training"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH ($PAIRS pairs)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

python rl/train_dpo_trl.py \
    --base_model "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 5e-7 \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --beta 0.1 \
    --lora_rank 64 \
    --warmup_ratio 0.1 \
    --max_seq_length 4096 \
    --wandb_run_name "dpo_experiment"

echo ""
echo "DPO training complete!"
echo "Run evaluation: bash scripts/03_evaluate.sh $OUTPUT_DIR/final_model"
