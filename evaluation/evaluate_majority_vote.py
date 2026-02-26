"""
多数投票评估脚本 (Majority Voting)
生成N个解答，取多数投票结果作为最终答案
可显著提升准确率 (+1-3%)
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional
from collections import Counter

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


def majority_vote(answers: list, gold: str) -> tuple:
    """
    Given a list of predicted answers, return (voted_answer, is_correct, vote_count).
    Normalizes answers for comparison using answers_match.
    """
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None, False, 0

    # Group answers that match each other
    groups = []
    for ans in valid_answers:
        found = False
        for group in groups:
            if answers_match(ans, group[0]):
                group.append(ans)
                found = True
                break
        if not found:
            groups.append([ans])

    # Find the largest group
    groups.sort(key=len, reverse=True)
    best_group = groups[0]
    voted_answer = best_group[0]
    vote_count = len(best_group)
    is_correct = answers_match(voted_answer, gold)

    return voted_answer, is_correct, vote_count


def evaluate_majority_vote(
    model_path: str,
    data_path: str,
    output_path: str,
    num_votes: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_samples: int = None,
    gpu_memory_utilization: float = 0.90,
):
    """Majority voting evaluation with vLLM"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"=== Majority Voting Evaluation (N={num_votes}) ===")
    print(f"Model: {model_path}")
    print(f"Temperature: {temperature}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        max_model_len=max_tokens + 512,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    eval_data = load_eval_data(data_path)
    if max_samples:
        eval_data = eval_data[:max_samples]

    dataset_name = eval_data[0].get("dataset", "") if eval_data else ""
    print(f"Dataset: {dataset_name}, Samples: {len(eval_data)}, Total generations: {len(eval_data) * num_votes}")

    # Build prompts - each problem repeated num_votes times
    all_prompts = []
    prompt_indices = []  # maps each prompt to its eval_data index
    for idx, item in enumerate(eval_data):
        messages = format_prompt(item["problem"], dataset_name)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for _ in range(num_votes):
            all_prompts.append(text)
            prompt_indices.append(idx)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    print(f"Generating {len(all_prompts)} completions...")
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"Generation done in {elapsed:.1f}s ({elapsed/len(all_prompts):.2f}s per completion)")

    # Group outputs by problem
    problem_outputs = [[] for _ in range(len(eval_data))]
    for i, output in enumerate(outputs):
        idx = prompt_indices[i]
        response = output.outputs[0].text
        predicted = extract_answer_from_solution(response)
        problem_outputs[idx].append({
            "response": response,
            "predicted": predicted,
        })

    # Majority vote
    correct = 0
    correct_greedy = 0  # first response as pseudo-greedy
    total = len(eval_data)
    results = []

    for i, item in enumerate(eval_data):
        gold = item["answer"]
        predictions = [o["predicted"] for o in problem_outputs[i]]

        voted_answer, is_correct, vote_count = majority_vote(predictions, gold)

        # Track first-response accuracy for comparison
        first_pred = predictions[0]
        first_correct = answers_match(first_pred, gold) if first_pred else False

        if is_correct:
            correct += 1
        if first_correct:
            correct_greedy += 1

        results.append({
            "problem": item["problem"],
            "gold_answer": gold,
            "voted_answer": str(voted_answer),
            "is_correct": is_correct,
            "vote_count": vote_count,
            "num_votes": num_votes,
            "all_predictions": [str(p) for p in predictions],
            "first_response_correct": first_correct,
            "level": item.get("level", ""),
            "dataset": item.get("dataset", ""),
        })

    accuracy_mv = correct / total if total > 0 else 0
    accuracy_single = correct_greedy / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Majority Vote (N={num_votes}): {correct}/{total} = {accuracy_mv*100:.2f}%")
    print(f"Single Response:   {correct_greedy}/{total} = {accuracy_single*100:.2f}%")
    print(f"MV Boost: +{(accuracy_mv - accuracy_single)*100:.2f}%")
    print(f"{'='*60}")

    # Level breakdown
    levels = set(r.get("level", "") for r in results if r.get("level"))
    if levels:
        print("\nBy level (MV):")
        for level in sorted(levels):
            lr = [r for r in results if r.get("level") == level]
            lc = sum(1 for r in lr if r["is_correct"])
            lt = len(lr)
            if lt > 0:
                print(f"  Level {level}: {lc}/{lt} = {lc/lt*100:.2f}%")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = {
        "model_path": model_path,
        "dataset": dataset_name,
        "accuracy_majority_vote": accuracy_mv,
        "accuracy_single": accuracy_single,
        "mv_boost": accuracy_mv - accuracy_single,
        "num_votes": num_votes,
        "temperature": temperature,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Majority Voting Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=8, help="Number of votes per problem")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    args = parser.parse_args()

    evaluate_majority_vote(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        num_votes=args.num_votes,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
