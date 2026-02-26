"""
SFT数据预处理脚本
将NuminaMath-CoT数据集转换为Qwen2.5模型的对话格式
"""
import os
import json
import random
import pandas as pd
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = os.environ.get("MATHMODEL_DATA_ROOT", "/tmp/dataset/mathmodel-dataset")
OUTPUT_DIR = os.environ.get(
    "MATHMODEL_PROCESSED_DIR",
    str(PROJECT_ROOT / "data_processing" / "processed"),
)

SYSTEM_PROMPT = (
    "You are a math problem solver. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)


def load_numina_cot(max_samples: int = None, seed: int = 42) -> list:
    """
    加载NuminaMath-CoT训练数据
    
    Args:
        max_samples: 最大样本数，None表示加载全部
        seed: 随机种子
    """
    data_dir = os.path.join(DATA_ROOT, "NuminaMath-CoT/data")
    all_data = []
    
    for i in range(5):
        path = os.path.join(data_dir, f"train-0000{i}-of-00005.parquet")
        df = pd.read_parquet(path)
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"NuminaMath-CoT 总样本数: {len(df)}")
    print(f"来源分布:\n{df['source'].value_counts().head(10)}")
    
    if max_samples and max_samples < len(df):
        random.seed(seed)
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"采样后样本数: {len(df)}")
    
    return df


def convert_to_chat_format(df: pd.DataFrame) -> list:
    """
    将NuminaMath-CoT转换为Qwen2.5对话格式
    
    格式:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "解答"}
    ]
    """
    records = []
    for _, row in df.iterrows():
        problem = row['problem']
        solution = row['solution']
        
        # 跳过空问题或解答
        if not problem or not solution or pd.isna(problem) or pd.isna(solution):
            continue
        
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "source": row.get('source', 'unknown'),
        }
        records.append(record)
    
    return records


def save_data(records: list, output_path: str):
    """保存为JSONL格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"保存 {len(records)} 条数据到 {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 准备不同规模的SFT数据集，用于消融实验
    sizes = {
        "sft_10k": 10000,
        "sft_50k": 50000,
        "sft_100k": 100000,
        "sft_full": None,  # 全量
    }
    
    # 先加载全量数据
    df_full = load_numina_cot()
    records_full = convert_to_chat_format(df_full)
    
    for name, size in sizes.items():
        if size is None:
            records = records_full
        else:
            random.seed(42)
            records = random.sample(records_full, min(size, len(records_full)))
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        save_data(records, output_path)
    
    print("\nSFT数据预处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
