#!/bin/bash
# =============================================================================
# Evaluate model on ALL benchmarks (greedy decoding)
# Benchmarks: MATH-500, GSM8K, AIME2024, AIME2025, MMLU-STEM, OlympiadBench
# Usage: bash scripts/05_evaluate_all_benchmarks.sh <model_path>
# =============================================================================
set -e

MODEL_PATH="${1:?Usage: $0 <model_path>}"
MODEL_NAME=$(basename "$MODEL_PATH")

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

cd "$(dirname "$0")/.."

EVAL_DIR="data_processing/processed/eval"
OUT_DIR="outputs/eval"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Evaluating on ALL benchmarks"
echo "  Model: $MODEL_PATH"
echo "============================================"

BENCHMARKS=("math500" "gsm8k" "aime2024" "aime2025" "mmlu_stem" "olympiadbench")
DISPLAY_NAMES=("MATH-500" "GSM8K" "AIME 2024" "AIME 2025" "MMLU-STEM" "OlympiadBench")

for i in "${!BENCHMARKS[@]}"; do
    BM="${BENCHMARKS[$i]}"
    NAME="${DISPLAY_NAMES[$i]}"
    DATA="$EVAL_DIR/${BM}.jsonl"
    OUT="$OUT_DIR/eval_${MODEL_NAME}_${BM}.json"
    
    if [ ! -f "$DATA" ]; then
        echo "⚠️  Skipping $NAME (data not found: $DATA)"
        continue
    fi
    
    echo ""
    echo "--- $NAME ---"
    python evaluation/evaluate_trl.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA" \
        --output_path "$OUT" \
        --max_tokens 3072 \
        --temperature 0.0 2>&1 | tail -3
done

echo ""
echo "============================================"
echo "Summary: $MODEL_NAME"
echo "============================================"
printf "%-20s %s\n" "Benchmark" "Accuracy"
printf "%-20s %s\n" "--------" "--------"
for i in "${!BENCHMARKS[@]}"; do
    BM="${BENCHMARKS[$i]}"
    NAME="${DISPLAY_NAMES[$i]}"
    OUT="$OUT_DIR/eval_${MODEL_NAME}_${BM}.json"
    if [ -f "$OUT" ]; then
        ACC=$(python -c "import json; d=json.load(open('$OUT')); print(f'{d[\"summary\"][\"accuracy\"]*100:.2f}%')" 2>/dev/null)
        printf "%-20s %s\n" "$NAME" "$ACC"
    fi
done
