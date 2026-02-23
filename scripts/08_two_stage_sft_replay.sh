#!/bin/bash
# =============================================================================
# Two-stage SFT with Replay mechanism
# Stage 1: Base math (OpenThoughts 82K)
# Stage 2: Competition math (NuminaMath-Competition 20K) + OT replay (20K)
#
# The replay mechanism prevents catastrophic forgetting when learning
# new domains by mixing in data from previous stages.
#
# Expected: GSM8K ~73%, MATH-500 ~45%
# =============================================================================
set -e

BASE_MODEL="${BASE_MODEL:-/tmp/pretrainmodel/Qwen2.5-3B}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate torch
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mathmodel-reasoning}"

cd "$(dirname "$0")/.."

# ---- Stage 1: Base Math SFT ----
echo "============================================"
echo "Stage 1: Base Math SFT (OpenThoughts 82K)"
echo "============================================"

STAGE1_DATA="${STAGE1_DATA:-data_processing/processed/sft_openthoughts_82k.jsonl}"
STAGE1_OUT="outputs/sft/sft_two_stage_s1"

if [ ! -f "$STAGE1_DATA" ]; then
    echo "Preparing Stage 1 data from OpenThoughts-114K (math subset)..."
    python -c "
import json, pandas as pd, os, random
random.seed(42)

data_dir = '/tmp/dataset/mathmodel-dataset/OpenThoughts-114k/data'
all_data = []
for i in range(6):
    path = os.path.join(data_dir, f'train-0000{i}-of-00006.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        all_data.append(df)

df = pd.concat(all_data)
math_df = df[df['domain'].str.contains('math', case=False, na=False)]
print(f'Math subset: {len(math_df)} samples')

records = []
for _, row in math_df.iterrows():
    records.append({
        'messages': [
            {'role': 'system', 'content': 'You are a math problem solver. Please reason step by step, and put your final answer within \\\\boxed{}.'},
            {'role': 'user', 'content': row['problem']},
            {'role': 'assistant', 'content': row['deepseek_solution']}
        ],
        'source': 'openthoughts'
    })

random.shuffle(records)
records = records[:82000]
os.makedirs('data_processing/processed', exist_ok=True)
with open('$STAGE1_DATA', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Saved {len(records)} samples')
"
fi

python sft/train_sft_fast.py \
    --base_model "$BASE_MODEL" \
    --data_path "$STAGE1_DATA" \
    --output_dir "$STAGE1_OUT" \
    --wandb_run_name "sft_two_stage_s1" \
    --lora_rank 256 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --warmup_ratio 0.1 \
    --save_steps 500

# Merge LoRA weights
echo "Merging Stage 1 LoRA weights..."
python utils/merge_lora_trl.py \
    --base_model "$BASE_MODEL" \
    --lora_model "$STAGE1_OUT/final_model" \
    --output_dir "$STAGE1_OUT/merged"

# ---- Stage 2: Competition Math + Replay ----
echo ""
echo "============================================"
echo "Stage 2: Competition + Replay (40K)"
echo "  Competition: 20K (NuminaMath competition)"
echo "  Replay: 20K (random OT from stage 1)"
echo "============================================"

STAGE2_DATA="${STAGE2_DATA:-data_processing/processed/sft_stage2_replay.jsonl}"
STAGE2_OUT="outputs/sft/sft_two_stage_s2"

if [ ! -f "$STAGE2_DATA" ]; then
    echo "Preparing Stage 2 data (competition + replay)..."
    python -c "
import json, random
random.seed(42)

# Load competition data (NuminaMath competition subset)
comp_data = []
with open('data_processing/processed/sft_competition_20k.jsonl') as f:
    for line in f:
        comp_data.append(json.loads(line))
print(f'Competition: {len(comp_data)} samples')

# Load OT data for replay
ot_data = []
with open('$STAGE1_DATA') as f:
    for line in f:
        ot_data.append(json.loads(line))
random.shuffle(ot_data)
replay_data = ot_data[:20000]
print(f'Replay: {len(replay_data)} samples')

# Combine and shuffle
combined = comp_data + replay_data
random.shuffle(combined)
with open('$STAGE2_DATA', 'w') as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Stage 2 data: {len(combined)} samples')
"
fi

python sft/train_sft_fast.py \
    --base_model "$STAGE1_OUT/merged" \
    --data_path "$STAGE2_DATA" \
    --output_dir "$STAGE2_OUT" \
    --wandb_run_name "sft_two_stage_s2_replay" \
    --lora_rank 256 \
    --num_epochs 1 \
    --learning_rate 2e-6 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --warmup_ratio 0.1 \
    --save_steps 200

echo ""
echo "Two-stage SFT with Replay complete!"
echo "Run evaluation: bash scripts/03_evaluate.sh $STAGE2_OUT/final_model"
