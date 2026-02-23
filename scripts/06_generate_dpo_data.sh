#!/bin/bash
# =============================================================================
# Generate DPO preference pairs from a trained model
# For each problem, generate N solutions, create preference pairs from
# correct (chosen) vs incorrect (rejected) solutions
# Usage: bash scripts/06_generate_dpo_data.sh <model_path>
# =============================================================================
set -e

MODEL_PATH="${1:?Usage: $0 <model_path>}"
OUTPUT_PATH="${2:-data_processing/processed/dpo_pairs.jsonl}"
NUM_SOLUTIONS="${3:-8}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

cd "$(dirname "$0")/.."

echo "============================================"
echo "Generating DPO Preference Pairs"
echo "  Model: $MODEL_PATH"
echo "  Solutions per problem: $NUM_SOLUTIONS"
echo "  Output: $OUTPUT_PATH"
echo "============================================"

python data_processing/generate_dpo_data_trl.py \
    --model_path "$MODEL_PATH" \
    --data_path data_processing/processed/rl_50k.jsonl \
    --output_path "$OUTPUT_PATH" \
    --num_solutions "$NUM_SOLUTIONS" \
    --max_tokens 2048 \
    --temperature 0.7 \
    --max_pairs_per_problem 3

PAIRS=$(wc -l < "$OUTPUT_PATH")
echo ""
echo "Generated $PAIRS preference pairs → $OUTPUT_PATH"
echo "Run DPO training: bash scripts/07_train_dpo.sh $MODEL_PATH $OUTPUT_PATH"
