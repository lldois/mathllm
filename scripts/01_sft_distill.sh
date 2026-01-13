#!/bin/bash
# =============================================================================
# Stage 1: SFT on Distilled Data (28K samples from Math-7B-Instruct)
# Expected: ~59.80% MATH-500 at checkpoint-3000
# Training time: ~5 hours on A100/C500 (64GB VRAM)
# =============================================================================
set -e

# ---- Configuration ----
BASE_MODEL="${BASE_MODEL:-/tmp/pretrainmodel/Qwen2.5-3B}"
DATA_PATH="${DATA_PATH:-data_processing/processed/sft_combined_v5.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/sft_distill}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft_distill}"

# ---- Environment ----
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mathmodel-reasoning}"

cd "$(dirname "$0")/.."

# ---- Validate ----
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: SFT data not found at $DATA_PATH"
    echo "Run data processing pipeline first (see README.md)"
    exit 1
fi

SAMPLES=$(wc -l < "$DATA_PATH")
echo "============================================"
echo "Stage 1: SFT on Distilled Data"
echo "  Base model: $BASE_MODEL"
echo "  Data: $DATA_PATH ($SAMPLES samples)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# ---- Train ----
python sft/train_sft_fast.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 4096 \
    --warmup_ratio 0.03 \
    --save_steps 500

echo ""
echo "SFT training complete!"
echo "Best checkpoint expected at: $OUTPUT_DIR/checkpoint-3000"
echo "Run evaluation: bash scripts/03_evaluate.sh $OUTPUT_DIR/checkpoint-3000"
