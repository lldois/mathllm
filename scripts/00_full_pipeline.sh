#!/bin/bash
# =============================================================================
# Full Pipeline: Data Processing → SFT → GRPO → Evaluation
# Reproduces the best model from scratch
# =============================================================================
set -e

echo "======================================================="
echo "  MathModel: Full Training Pipeline"
echo "  Qwen2.5-3B-Base → SFT → GRPO → 61.40% MATH-500"
echo "======================================================="

cd "$(dirname "$0")/.."

# ---- Step 0: Data Processing ----
echo ""
echo "[Step 0/5] Preparing data..."
python data_processing/prepare_sft_data.py
python data_processing/prepare_rl_data.py
python data_processing/prepare_eval_data.py

# Distillation (requires Math-7B-Instruct model)
if [ -d "/tmp/pretrainmodel/Qwen2.5-Math-7B-Instruct" ]; then
    echo "Running teacher distillation (Math-7B-Instruct → solutions)..."
    python data_processing/distill_math7b_large.py \
        --data_path data_processing/processed/rl_50k.jsonl \
        --output_path data_processing/processed/sft_distill_math7b.jsonl \
        --max_problems 50000
    python data_processing/enhanced_distill_targeted.py \
        --rl_data data_processing/processed/rl_50k.jsonl \
        --sft_data data_processing/processed/sft_distill_math7b.jsonl \
        --output_path data_processing/processed/sft_enhanced_distill.jsonl \
        --max_problems 0
    python data_processing/combine_sft_data.py \
        --inputs data_processing/processed/sft_distill_math7b.jsonl \
                 data_processing/processed/sft_enhanced_distill.jsonl \
        --output data_processing/processed/sft_combined_v5.jsonl
else
    echo "WARNING: Math-7B-Instruct not found, using existing SFT data"
fi

# ---- Step 1: SFT Training ----
echo ""
echo "[Step 1/5] SFT training on distilled data..."
bash scripts/01_sft_distill.sh

# ---- Step 2: Evaluate SFT ----
echo ""
echo "[Step 2/5] Evaluating SFT model..."
SFT_MODEL="outputs/sft/sft_distill/checkpoint-3000"
bash scripts/03_evaluate.sh "$SFT_MODEL"

# ---- Step 3: GRPO Training ----
echo ""
echo "[Step 3/5] GRPO training..."
bash scripts/02_grpo.sh "$SFT_MODEL"

# ---- Step 4: Evaluate GRPO (greedy) ----
echo ""
echo "[Step 4/5] Evaluating GRPO model (greedy)..."
GRPO_MODEL="outputs/grpo/grpo_from_sft/checkpoint-300"
bash scripts/05_evaluate_all_benchmarks.sh "$GRPO_MODEL"

# ---- Step 5: Evaluate GRPO (MV@64) ----
echo ""
echo "[Step 5/5] Evaluating GRPO model (Majority Vote@64)..."
bash scripts/04_evaluate_mv.sh "$GRPO_MODEL"

echo ""
echo "======================================================="
echo "  Pipeline Complete!"
echo "  Best model: $GRPO_MODEL"
echo "======================================================="
