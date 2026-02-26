"""
评估数据预处理脚本
统一格式化所有评估数据集
"""
import os
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = os.environ.get("MATHMODEL_DATA_ROOT", "/tmp/dataset/mathmodel-dataset")
OUTPUT_DIR = os.environ.get(
    "MATHMODEL_EVAL_DIR",
    str(PROJECT_ROOT / "data_processing" / "processed" / "eval"),
)


def prepare_math500():
    """MATH-500 评估集"""
    path = os.path.join(DATA_ROOT, "MATH-500/test.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            records.append({
                "problem": item["problem"],
                "answer": item["answer"],
                "level": item.get("level", ""),
                "subject": item.get("subject", ""),
                "dataset": "MATH-500",
                "unique_id": item.get("unique_id", ""),
            })
    return records


def prepare_gsm8k():
    """GSM8K 测试集 - 从HuggingFace数据脚本中获取URL后本地加载"""
    # gsm8k是脚本形式的数据集，需要通过datasets库加载
    # 先尝试从本地缓存加载
    try:
        from datasets import load_dataset
        ds = load_dataset(os.path.join(DATA_ROOT, "gsm8k"), "main", split="test",
                         trust_remote_code=True)
        records = []
        for item in ds:
            # 从答案中提取数字
            answer_text = item["answer"]
            # GSM8K答案格式: "解题过程\n#### 数字答案"
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()
            else:
                final_answer = answer_text
            records.append({
                "problem": item["question"],
                "answer": final_answer,
                "level": "grade_school",
                "subject": "math",
                "dataset": "GSM8K",
                "unique_id": f"gsm8k_{len(records)}",
            })
        return records
    except Exception as e:
        print(f"加载GSM8K出错: {e}")
        print("尝试直接下载...")
        return []


def prepare_aime2024():
    """AIME 2024 评估集"""
    path = os.path.join(DATA_ROOT, "AIME_2024/aime_2024_problems.parquet")
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "problem": row["Problem"],
            "answer": str(row["Answer"]),
            "level": "competition",
            "subject": "math",
            "dataset": "AIME2024",
            "unique_id": row["ID"],
        })
    return records


def prepare_aime2025():
    """AIME 2025 评估集"""
    records = []
    for fname in ["aime2025-I.jsonl", "aime2025-II.jsonl"]:
        path = os.path.join(DATA_ROOT, "AIME2025", fname)
        with open(path) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                part = "I" if "I.jsonl" in fname else "II"
                records.append({
                    "problem": item["question"],
                    "answer": str(item["answer"]),
                    "level": "competition",
                    "subject": "math",
                    "dataset": "AIME2025",
                    "unique_id": f"aime2025_{part}_{i}",
                })
    return records


def prepare_olympiadbench():
    """OlympiadBench 评估集（数学部分）"""
    ob_dir = os.path.join(DATA_ROOT, "OlympiadBench/OlympiadBench")
    records = []
    # 只加载数学的text-only部分
    for subdir in sorted(os.listdir(ob_dir)):
        if 'maths' in subdir and 'TO' in subdir:  # Text-Only maths
            parquet_dir = os.path.join(ob_dir, subdir)
            for f in os.listdir(parquet_dir):
                if f.endswith('.parquet'):
                    df = pd.read_parquet(os.path.join(parquet_dir, f))
                    for _, row in df.iterrows():
                        answer = str(row.get('final_answer', row.get('answer', '')))
                        if not answer or answer == 'nan':
                            continue
                        records.append({
                            "problem": row.get('question', row.get('problem', '')),
                            "answer": answer,
                            "level": "olympiad",
                            "subject": "math",
                            "dataset": "OlympiadBench",
                            "unique_id": f"ob_{subdir}_{len(records)}",
                        })
    return records


def prepare_mmlu_stem():
    """MMLU-STEM 评估集"""
    path = os.path.join(DATA_ROOT, "MMLU-STEM/stem.json")
    data = json.load(open(path))
    records = []
    for item in data:
        choices = item["choices"]
        answer_idx = item["answer"]
        answer_letter = chr(65 + answer_idx)  # 0->A, 1->B, ...
        # 构造选择题格式
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        problem = f"{item['question']}\n\n{choices_text}"
        records.append({
            "problem": problem,
            "answer": answer_letter,
            "level": "college",
            "subject": item.get("subject", "stem"),
            "dataset": "MMLU-STEM",
            "unique_id": f"mmlu_{len(records)}",
        })
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = {
        "math500": prepare_math500,
        "gsm8k": prepare_gsm8k,
        "aime2024": prepare_aime2024,
        "aime2025": prepare_aime2025,
        "olympiadbench": prepare_olympiadbench,
        "mmlu_stem": prepare_mmlu_stem,
    }
    
    for name, prepare_fn in datasets.items():
        print(f"\n处理 {name}...")
        try:
            records = prepare_fn()
            if records:
                output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
                with open(output_path, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  保存 {len(records)} 条数据到 {output_path}")
            else:
                print(f"  警告: {name} 无数据")
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n评估数据预处理完成!")


if __name__ == "__main__":
    main()
