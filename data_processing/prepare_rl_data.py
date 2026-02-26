"""
RL训练数据预处理脚本
将OpenThoughts-114k转换为GRPO训练所需的prompt格式
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


def load_openthoughts(max_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """加载OpenThoughts-114k数据"""
    data_dir = os.path.join(DATA_ROOT, "OpenThoughts-114k/data")
    all_data = []
    
    for i in range(6):
        path = os.path.join(data_dir, f"train-0000{i}-of-00006.parquet")
        df = pd.read_parquet(path)
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"OpenThoughts-114k 总样本数: {len(df)}")
    
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"采样后样本数: {len(df)}")
    
    return df


def extract_answer_from_conversations(conversations: list) -> str:
    """从对话中提取标准答案"""
    for conv in conversations:
        if conv['from'] == 'assistant':
            text = conv['value']
            # 从solution部分提取boxed答案
            import re
            # 匹配 \boxed{...}
            pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()
    return ""


def convert_to_rl_format(df: pd.DataFrame) -> list:
    """
    转换为GRPO训练格式
    每条数据包含: prompt (对话格式) + answer (标准答案)
    """
    records = []
    for _, row in df.iterrows():
        conversations = row['conversations']
        
        # 提取用户问题
        user_msg = None
        for conv in conversations:
            if conv['from'] == 'user':
                user_msg = conv['value']
                break
        
        if not user_msg:
            continue
        
        # 提取标准答案
        answer = extract_answer_from_conversations(conversations)
        if not answer:
            continue
        
        record = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "answer": answer,
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
    
    # 加载全量数据
    df = load_openthoughts()
    records = convert_to_rl_format(df)
    print(f"有效RL训练数据: {len(records)} 条")
    
    # 保存不同规模的数据集
    sizes = {
        "rl_20k": 20000,
        "rl_50k": 50000,
        "rl_full": None,
    }
    
    for name, size in sizes.items():
        if size is None:
            subset = records
        else:
            random.seed(42)
            subset = random.sample(records, min(size, len(records)))
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        save_data(subset, output_path)
    
    print("\nRL数据预处理完成!")


if __name__ == "__main__":
    main()
