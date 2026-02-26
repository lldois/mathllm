"""
评估脚本 (TRL版本)
使用原生vLLM加速推理，不依赖unsloth
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.answer_extraction import extract_answer_from_solution, answers_match, extract_boxed_answer

SYSTEM_PROMPT = (
    "You are a math problem solver. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)


def load_eval_data(data_path: str) -> list:
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def format_prompt(problem: str, dataset_name: str = "") -> list:
    if dataset_name == "MMLU-STEM":
        system = "You are a helpful assistant. Choose the correct answer from the options. Put your answer (A, B, C, or D) within \\boxed{}."
    else:
        system = SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]


def evaluate_with_vllm(
    model_path: str,
    data_path: str,
    output_path: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_samples: int = None,
    lora_path: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
):
    """使用原生vLLM进行批量评估"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else model_path,
        trust_remote_code=True,
    )

    llm_kwargs = dict(
        model=model_path,
        tokenizer=lora_path if lora_path else model_path,
        max_model_len=max_tokens + 512,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    if lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 256
        print(f"加载LoRA: {lora_path}")

    llm = LLM(**llm_kwargs)

    # 加载评估数据
    eval_data = load_eval_data(data_path)
    if max_samples:
        eval_data = eval_data[:max_samples]

    dataset_name = eval_data[0].get("dataset", "") if eval_data else ""
    print(f"评估数据集: {dataset_name}, 样本数: {len(eval_data)}")

    # 构造prompts
    prompts = []
    for item in eval_data:
        messages = format_prompt(item["problem"], dataset_name)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
    )

    print(f"开始推理... (样本数={len(prompts)})")
    start_time = time.time()

    if lora_path:
        lora_request = LoRARequest("eval_lora", 1, lora_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - start_time
    print(f"推理完成，用时: {elapsed:.1f}s")

    # 评估结果
    correct = 0
    total = len(eval_data)
    results = []

    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        response = output.outputs[0].text
        predicted = extract_answer_from_solution(response)
        gold = item["answer"]
        is_correct = answers_match(predicted, gold) if predicted else False

        if is_correct:
            correct += 1

        results.append({
            "problem": item["problem"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "response": response,
            "level": item.get("level", ""),
            "dataset": item.get("dataset", ""),
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"准确率: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # 按level统计
    levels = set(r.get("level", "") for r in results if r.get("level"))
    if levels:
        print("\n按难度级别统计:")
        for level in sorted(levels):
            level_results = [r for r in results if r.get("level") == level]
            level_correct = sum(1 for r in level_results if r["is_correct"])
            level_total = len(level_results)
            if level_total > 0:
                print(f"  Level {level}: {level_correct}/{level_total} = {level_correct/level_total:.4f}")

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = {
        "model_path": model_path,
        "lora_path": lora_path,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果保存到: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="数学推理模型评估 (TRL版)")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA适配器路径")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据路径")
    parser.add_argument("--output_path", type=str, required=True, help="结果输出路径")
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    args = parser.parse_args()

    evaluate_with_vllm(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        lora_path=args.lora_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
