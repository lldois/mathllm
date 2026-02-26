#!/usr/bin/env python3
"""
Enhanced multi-sample distillation: generate solutions for problems NOT covered
by existing SFT data. Uses our best model OR Math-7B for diversity.
Outputs in SFT training format.
"""
import json
import argparse
import random
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.answer_extraction import extract_boxed_answer, extract_answer_from_solution, normalize_answer, answers_match

SYSTEM_PROMPT = (
    "You are a math problem solver. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)


def extract_problem_from_sft_record(record):
    """Extract problem text from any SFT record format used in this repo."""
    problem = record.get("problem", "")
    if problem:
        return problem

    messages = record.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") == "user" and message.get("content"):
                return message["content"]

    text = record.get("text", "")
    if text and "<|im_start|>user\n" in text:
        return text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0].strip()

    return ""


def load_rl_problems_not_in_sft(rl_path, sft_path, max_problems=0):
    """Load RL problems that don't have solutions in the SFT data"""
    sft_problems = set()
    with open(sft_path) as f:
        for line in f:
            d = json.loads(line)
            problem_text = extract_problem_from_sft_record(d)
            if problem_text:
                sft_problems.add(problem_text.strip()[:300])
    print(f"SFT data has {len(sft_problems)} unique problems")

    unsolved = []
    total = 0
    with open(rl_path) as f:
        for line in f:
            d = json.loads(line)
            total += 1
            prompt = d.get('prompt', [])
            problem_text = ""
            for m in prompt:
                if m.get('role') == 'user':
                    problem_text = m['content']
                    break
            answer = d.get('answer', '')
            if problem_text and problem_text.strip()[:300] not in sft_problems:
                unsolved.append({"problem": problem_text, "answer": str(answer)})

    print(f"RL data: {total} total, {len(unsolved)} not in SFT data")

    if max_problems > 0 and len(unsolved) > max_problems:
        random.seed(42)
        unsolved = random.sample(unsolved, max_problems)
        print(f"Sampled {max_problems} problems")

    return unsolved


def generate_solutions(model_path, problems, num_solutions=2, max_tokens=3072, temperature=0.7, batch_size=256):
    """Generate solutions using vLLM with batching for memory efficiency"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        n=num_solutions,
    )

    # Build all prompts
    prompts = []
    for prob in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prob["problem"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f"Generating {num_solutions} solutions for {len(prompts)} problems...")
    print(f"Total generations: {len(prompts) * num_solutions}")

    # Generate in batches for progress reporting
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(batch_outputs)
        done = min(i + batch_size, len(prompts))
        print(f"  Progress: {done}/{len(prompts)} problems ({done/len(prompts)*100:.1f}%)")

    return all_outputs


def filter_and_save(problems, outputs, output_path, keep_all_correct=False):
    """Filter correct solutions and save as SFT data"""
    total_correct = 0
    total_generated = 0
    problems_with_correct = 0
    sft_data = []

    for prob, output in zip(problems, outputs):
        correct_solutions = []
        for gen in output.outputs:
            total_generated += 1
            sol_text = gen.text

            # Try both extraction methods
            extracted = extract_boxed_answer(sol_text)
            if not extracted:
                extracted = extract_answer_from_solution(sol_text)

            if extracted and prob["answer"]:
                if answers_match(extracted, prob["answer"]):
                    total_correct += 1
                    correct_solutions.append(sol_text)

        if correct_solutions:
            problems_with_correct += 1
            if keep_all_correct:
                for sol in correct_solutions:
                    sft_data.append(_make_sft_item(prob, sol))
            else:
                best = min(correct_solutions, key=len)
                sft_data.append(_make_sft_item(prob, best))

    print(f"\n=== Distillation Results ===")
    print(f"Problems: {len(problems)}")
    print(f"Generated: {total_generated}")
    print(f"Correct: {total_correct} ({total_correct/max(total_generated,1)*100:.1f}%)")
    print(f"Problems with ≥1 correct: {problems_with_correct}/{len(problems)} ({problems_with_correct/max(len(problems),1)*100:.1f}%)")
    print(f"SFT samples: {len(sft_data)}")

    random.shuffle(sft_data)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to: {output_path}")

    return sft_data


def _make_sft_item(prob, solution):
    return {
        "text": "",  # will be filled by SFT script
        "problem": prob["problem"],
        "answer": prob["answer"],
        "solution": solution,
        "source": "enhanced_distill",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prob["problem"]},
            {"role": "assistant", "content": solution}
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/tmp/pretrainmodel/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--rl_data", default="data_processing/processed/rl_50k.jsonl")
    parser.add_argument("--sft_data", default="data_processing/processed/sft_distill_math7b.jsonl")
    parser.add_argument("--output_path", default="data_processing/processed/sft_enhanced_distill.jsonl")
    parser.add_argument("--max_problems", type=int, default=0,
                        help="Max problems to distill (0=all unsolved)")
    parser.add_argument("--num_solutions", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--keep_all_correct", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    problems = load_rl_problems_not_in_sft(args.rl_data, args.sft_data, args.max_problems)

    if not problems:
        print("No unsolved problems found!")
        sys.exit(1)

    outputs = generate_solutions(
        args.model_path, problems,
        num_solutions=args.num_solutions,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    filter_and_save(problems, outputs, args.output_path, args.keep_all_correct)
