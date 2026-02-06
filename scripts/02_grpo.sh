#!/bin/bash
# =============================================================================
# Stage 2: GRPO from SFT checkpoint
# Expected: ~61.40% MATH-500 greedy at checkpoint-300
# Training time: ~3 hours on A100/C500 (64GB VRAM)
# =============================================================================
set -e

# ---- Configuration ----
SFT_MODEL="${1:-outputs/sft/sft_distill/checkpoint-3000}"
DATA_PATH="${DATA_PATH:-data_processing/processed/rl_50k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo/grpo_from_sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_from_sft}"

# ---- Environment ----
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mathmodel-reasoning}"

cd "$(dirname "$0")/.."

# ---- Validate ----
if [ ! -d "$SFT_MODEL" ]; then
    echo "ERROR: SFT model not found at $SFT_MODEL"
    echo "Run Stage 1 first: bash scripts/01_sft_distill.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: RL data not found at $DATA_PATH"
    exit 1
fi

echo "============================================"
echo "Stage 2: GRPO Reinforcement Learning"
echo "  SFT model: $SFT_MODEL"
echo "  RL data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Loss: DAPO | G=32 | lr=3e-7 constant"
echo "============================================"

# ---- Train ----
python rl/train_grpo_v2_trl.py \
    --model_path "$SFT_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$WANDB_RUN_NAME" \
    --max_steps 300 \
    --learning_rate 3e-7 \
    --lr_scheduler constant \
    --loss_type dapo \
    --beta 0.0 \
    --epsilon 0.1 \
    --epsilon_high 0.28 \
    --mask_truncated \
    --temperature 0.9 \
    --correctness_only \
    --num_generations 32 \
    --generation_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --save_steps 25

echo ""
echo "GRPO training complete!"
echo "Best checkpoint expected at: $OUTPUT_DIR/checkpoint-300"
echo "Run evaluation: bash scripts/03_evaluate.sh $OUTPUT_DIR/checkpoint-300"
