"""
Weighted Majority Voting with log-probability confidence.
Uses vLLM's n parameter for efficient multi-sample generation.
Weights votes by exp(mean_logprob) for better answer selection.
"""
import os
import sys
import json
import argparse
import time
import math
from pathlib import Path
from collections import defaultdict

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


def format_prompt(problem: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def weighted_majority_vote(answers_with_logprobs: list, gold: str):
    """
    Weighted majority vote using log-probability confidence.
    answers_with_logprobs: list of (answer_str, mean_logprob)
    Returns: (voted_answer, is_correct, weight_sum, method_results)
    """
    # Group answers and accumulate weights
    answer_groups = defaultdict(lambda: {"weight": 0.0, "count": 0, "answers": []})
    
    for ans, logprob in answers_with_logprobs:
        if ans is None:
            continue
        # Use exp(mean_logprob) as weight (higher prob = higher weight)
        weight = math.exp(logprob) if logprob is not None else 1.0
        
        # Find matching group
        matched = False
        for key in list(answer_groups.keys()):
            if answers_match(ans, key):
                answer_groups[key]["weight"] += weight
                answer_groups[key]["count"] += 1
                answer_groups[key]["answers"].append(ans)
                matched = True
                break
        if not matched:
            answer_groups[ans]["weight"] += weight
            answer_groups[ans]["count"] += 1
            answer_groups[ans]["answers"].append(ans)
    
    if not answer_groups:
        return None, False, 0, {}
    
    # Weighted vote: pick answer with highest total weight
    weighted_best = max(answer_groups.keys(), key=lambda k: answer_groups[k]["weight"])
    weighted_correct = answers_match(weighted_best, gold)
    
    # Simple vote for comparison
    simple_best = max(answer_groups.keys(), key=lambda k: answer_groups[k]["count"])
    simple_correct = answers_match(simple_best, gold)
    
    method_results = {
        "weighted": {"answer": weighted_best, "correct": weighted_correct},
        "simple": {"answer": simple_best, "correct": simple_correct},
    }
    
    return weighted_best, weighted_correct, answer_groups[weighted_best]["weight"], method_results


def evaluate_weighted_mv(
    model_path: str,
    data_path: str,
    output_path: str,
    num_votes: int = 64,
    max_tokens: int = 3072,
    temperature: float = 0.7,
    max_samples: int = None,
    gpu_memory_utilization: float = 0.90,
):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"=== Weighted Majority Voting (N={num_votes}) ===")
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

    print(f"Samples: {len(eval_data)}, Total generations: {len(eval_data) * num_votes}")

    # Build prompts - one per problem, use n for multiple samples
    all_prompts = []
    for item in eval_data:
        messages = format_prompt(item["problem"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(text)

    # Use n parameter for efficient multi-sample generation + logprobs
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        n=num_votes,
        logprobs=1,  # request logprobs for confidence weighting
    )

    print(f"Generating {len(all_prompts)} prompts × {num_votes} samples...")
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - start_time
    total_gens = len(all_prompts) * num_votes
    print(f"Generation done in {elapsed:.1f}s ({elapsed/total_gens:.3f}s per completion)")

    # Process results
    correct_weighted = 0
    correct_simple = 0
    total = len(eval_data)
    results = []

    for i, (item, output) in enumerate(zip(eval_data, outputs)):
        gold = item["answer"]
        
        answers_with_logprobs = []
        for comp in output.outputs:
            response = comp.text
            predicted = extract_answer_from_solution(response)
            
            # Calculate mean log-prob from cumulative_logprob
            mean_logprob = None
            if hasattr(comp, 'cumulative_logprob') and comp.cumulative_logprob is not None:
                num_tokens = len(comp.token_ids) if hasattr(comp, 'token_ids') else 1
                if num_tokens > 0:
                    mean_logprob = comp.cumulative_logprob / num_tokens
            
            answers_with_logprobs.append((predicted, mean_logprob))
        
        voted_answer, is_correct, weight, method_results = weighted_majority_vote(
            answers_with_logprobs, gold
        )
        
        if is_correct:
            correct_weighted += 1
        if method_results.get("simple", {}).get("correct", False):
            correct_simple += 1

        results.append({
            "problem": item["problem"][:100],
            "gold_answer": gold,
            "weighted_answer": str(voted_answer),
            "simple_answer": str(method_results.get("simple", {}).get("answer", "")),
            "weighted_correct": is_correct,
            "simple_correct": method_results.get("simple", {}).get("correct", False),
            "num_votes": num_votes,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] Weighted: {correct_weighted/(i+1):.2%}, Simple: {correct_simple/(i+1):.2%}")

    acc_weighted = correct_weighted / total
    acc_simple = correct_simple / total

    summary = {
        "model": model_path,
        "num_votes": num_votes,
        "temperature": temperature,
        "total": total,
        "weighted_correct": correct_weighted,
        "weighted_accuracy": acc_weighted,
        "simple_correct": correct_simple,
        "simple_accuracy": acc_simple,
        "improvement": acc_weighted - acc_simple,
        "generation_time_s": elapsed,
    }

    output_data = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Results (N={num_votes}):")
    print(f"  Weighted MV: {acc_weighted:.2%} ({correct_weighted}/{total})")
    print(f"  Simple MV:   {acc_simple:.2%} ({correct_simple}/{total})")
    print(f"  Improvement: {acc_weighted - acc_simple:+.2%}")
    print(f"  Saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Weighted Majority Voting Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = parser.parse_args()

    evaluate_weighted_mv(
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
