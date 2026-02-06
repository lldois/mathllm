#!/bin/bash
# =============================================================================
# Evaluate model on MATH-500 with Majority Vote@64
# Usage: bash scripts/04_evaluate_mv.sh <model_path>
# =============================================================================
set -e

MODEL_PATH="${1:?Usage: $0 <model_path>}"
NUM_VOTES="${2:-64}"
OUTPUT_PATH="${3:-outputs/eval/eval_math500_mv${NUM_VOTES}_$(basename $MODEL_PATH).json}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

cd "$(dirname "$0")/.."

echo "============================================"
echo "Evaluating on MATH-500 (Majority Vote@$NUM_VOTES)"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_PATH"
echo "============================================"

python evaluation/evaluate_majority_vote.py \
    --model_path "$MODEL_PATH" \
    --data_path data_processing/processed/eval/math500.jsonl \
    --output_path "$OUTPUT_PATH" \
    --max_tokens 3072 \
    --num_votes "$NUM_VOTES" \
    --temperature 0.7

echo ""
echo "Result:"
python -c "import json; d=json.load(open('$OUTPUT_PATH')); print(f'  MATH-500 MV@$NUM_VOTES Accuracy: {d[\"summary\"][\"accuracy\"]*100:.2f}%')"
