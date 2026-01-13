#!/bin/bash
# =============================================================================
# Evaluate model on MATH-500 (greedy decoding)
# Usage: bash scripts/03_evaluate.sh <model_path>
# =============================================================================
set -e

MODEL_PATH="${1:?Usage: $0 <model_path>}"
OUTPUT_PATH="${2:-outputs/eval/eval_math500_$(basename $MODEL_PATH).json}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

cd "$(dirname "$0")/.."

echo "============================================"
echo "Evaluating on MATH-500 (greedy)"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_PATH"
echo "============================================"

python evaluation/evaluate_trl.py \
    --model_path "$MODEL_PATH" \
    --data_path data_processing/processed/eval/math500.jsonl \
    --output_path "$OUTPUT_PATH" \
    --max_tokens 3072 \
    --temperature 0.0

echo ""
echo "Result:"
python -c "import json; d=json.load(open('$OUTPUT_PATH')); print(f'  MATH-500 Accuracy: {d[\"summary\"][\"accuracy\"]*100:.2f}%')"
