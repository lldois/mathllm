"""Generate DPO preference pairs using vLLM batch inference.

For each problem, generates N solutions, then pairs correct (chosen)
with incorrect (rejected) solutions for DPO training.

Usage:
    python data_processing/generate_dpo_data_trl.py \
        --model_path outputs/grpo/grpo_exp132_g32_from_sft131/checkpoint-300 \
        --data_path data_processing/processed/rl_50k.jsonl \
        --output_path data_processing/processed/dpo_exp132ck300.jsonl \
        --n_solutions 8 --temperature 0.7
"""
import argparse
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.answer_extraction import extract_answer_from_solution, answers_match


def extract_problem_text(item):
    """Extract problem text from RL data format."""
    prompt = item.get("prompt", item.get("problem", ""))
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user":
                return msg["content"]
        return prompt[-1]["content"] if prompt else ""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_problems", type=int, default=0, help="0=all")
    parser.add_argument("--n_solutions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_pairs_per_problem", type=int, default=3,
                        help="Max DPO pairs per problem")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    problems = []
    with open(args.data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if args.max_problems > 0:
        random.seed(42)
        random.shuffle(problems)
        problems = problems[:args.max_problems]
    print(f"Loaded {len(problems)} problems")

    # Load vLLM
    print(f"Loading model from {args.model_path}...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 512,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_solutions,
    )

    all_pairs = []
    all_correct = []  # Also save correct solutions for STaR augmentation
    total_problems_with_pairs = 0
    start_time = time.time()

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    correct_path = args.output_path.replace('.jsonl', '_correct.jsonl')

    for batch_start in range(0, len(problems), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(problems))
        batch = problems[batch_start:batch_end]

        # Build prompts
        chat_prompts = []
        for item in batch:
            text = extract_problem_text(item)
            messages = [{"role": "user", "content": text}]
            chat_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        bt = time.time()
        outputs = llm.generate(chat_prompts, sampling_params)
        gen_time = time.time() - bt

        batch_pairs = 0
        batch_correct_count = 0
        for i, output in enumerate(outputs):
            item = batch[i]
            true_answer = item.get("answer", item.get("expected_answer", ""))
            problem_text = extract_problem_text(item)

            correct_solutions = []
            incorrect_solutions = []

            for completion in output.outputs:
                text = completion.text
                pred = extract_answer_from_solution(text)
                if pred and answers_match(pred, true_answer):
                    correct_solutions.append(text)
                else:
                    incorrect_solutions.append(text)

            # Save all correct solutions for SFT augmentation
            for sol in correct_solutions:
                sft_sample = {
                    "messages": [
                        {"role": "user", "content": problem_text},
                        {"role": "assistant", "content": sol}
                    ]
                }
                all_correct.append(sft_sample)
                batch_correct_count += 1

            # Create DPO pairs only when both exist
            if correct_solutions and incorrect_solutions:
                total_problems_with_pairs += 1
                n_pairs = min(args.max_pairs_per_problem,
                              len(correct_solutions), len(incorrect_solutions))
                random.shuffle(correct_solutions)
                random.shuffle(incorrect_solutions)

                for j in range(n_pairs):
                    pair = {
                        "prompt": [{"role": "user", "content": problem_text}],
                        "chosen": [{"role": "assistant", "content": correct_solutions[j]}],
                        "rejected": [{"role": "assistant", "content": incorrect_solutions[j]}],
                        "answer": true_answer,
                    }
                    all_pairs.append(pair)
                    batch_pairs += 1

        # Write incrementally
        with open(args.output_path, 'a') as f:
            for pair in all_pairs[len(all_pairs) - batch_pairs:]:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        with open(correct_path, 'a') as f:
            for sol in all_correct[len(all_correct) - batch_correct_count:]:
                f.write(json.dumps(sol, ensure_ascii=False) + '\n')

        elapsed = time.time() - start_time
        print(f"Batch {batch_start // args.batch_size + 1}: "
              f"{batch_pairs} pairs, {batch_correct_count} correct from {len(batch)} problems, "
              f"gen time: {gen_time:.0f}s, "
              f"total pairs: {len(all_pairs)}, correct solutions: {len(all_correct)}, "
              f"problems with pairs: {total_problems_with_pairs}")

    elapsed = time.time() - start_time
    print(f"\n=== DPO Generation Complete ===")
    print(f"Total problems: {len(problems)}")
    print(f"Problems with both correct & incorrect: {total_problems_with_pairs} "
          f"({total_problems_with_pairs / len(problems) * 100:.1f}%)")
    print(f"Total DPO pairs: {len(all_pairs)}")
    print(f"Total correct solutions: {len(all_correct)}")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"DPO pairs saved to: {args.output_path}")
    print(f"Correct solutions saved to: {correct_path}")


if __name__ == "__main__":
    main()
