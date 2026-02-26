"""
Large-scale distillation from Math-7B-Instruct on RL problems.
Generates greedy solutions, verifies correctness, saves correct ones for SFT.
"""
import os, sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.answer_extraction import extract_answer_from_solution, answers_match, extract_boxed_answer

SYSTEM_PROMPT = (
    "You are a math problem solver. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)


def load_rl_data(path, max_problems=None, skip_first=0):
    """Load RL data, extract problem text from prompt field."""
    problems = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip_first:
                continue
            if max_problems and len(problems) >= max_problems:
                break
            item = json.loads(line)
            prompt = item.get("prompt", item.get("problem", item.get("question", "")))
            if isinstance(prompt, list):
                # Chat messages format - extract user content
                text = ""
                for msg in prompt:
                    if msg.get("role") == "user":
                        text = msg["content"]
                        break
                if not text and prompt:
                    text = prompt[-1]["content"]
            else:
                text = prompt

            answer = item.get("answer", item.get("expected_answer", ""))
            if text and answer:
                problems.append({"problem": text, "answer": answer, "index": i})
    return problems


def build_chat_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/tmp/pretrainmodel/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--data_path", default="data_processing/processed/rl_50k.jsonl")
    parser.add_argument("--output_path", default="data_processing/processed/sft_distill_math7b.jsonl")
    parser.add_argument("--max_problems", type=int, default=50000)
    parser.add_argument("--skip_first", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--n_solutions", type=int, default=1, help="Number of solutions per problem (1=greedy)")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"Loading problems from {args.data_path}...")
    problems = load_rl_data(args.data_path, args.max_problems, args.skip_first)
    print(f"Loaded {len(problems)} problems")

    print(f"Loading model: {args.model_path}")
    llm = LLM(args.model_path, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_solutions,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing progress
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item.get("problem_index", -1))
        print(f"Found {len(existing)} existing solutions, resuming...")
        problems = [p for p in problems if p["index"] not in existing]
        print(f"Remaining: {len(problems)} problems")

    total_correct = len(existing)
    total_processed = len(existing)
    t0 = time.time()

    # Process in batches
    for batch_start in range(0, len(problems), args.batch_size):
        batch = problems[batch_start:batch_start + args.batch_size]
        batch_prompts = [build_chat_prompt(tokenizer, p["problem"]) for p in batch]

        bt0 = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        bt1 = time.time()

        batch_correct = 0
        with open(output_path, "a") as fout:
            for prob, output in zip(batch, outputs):
                for completion in output.outputs:
                    solution = completion.text
                    pred = extract_boxed_answer(solution)
                    if not pred:
                        pred = extract_answer_from_solution(solution)
                    if pred and answers_match(pred, prob["answer"]):
                        # Build SFT format
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prob["problem"]},
                            {"role": "assistant", "content": solution},
                        ]
                        record = {
                            "messages": messages,
                            "problem": prob["problem"],
                            "solution": solution,
                            "source": "math7b_distill",
                            "problem_index": prob["index"],
                            "answer": prob["answer"],
                            "predicted_answer": pred,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        batch_correct += 1

        total_correct += batch_correct
        total_processed += len(batch)
        elapsed = time.time() - t0
        rate = total_processed / elapsed * 3600

        print(f"Batch {batch_start//args.batch_size + 1}: "
              f"{batch_correct}/{len(batch)} correct ({batch_correct/len(batch)*100:.1f}%), "
              f"batch time: {bt1-bt0:.0f}s, "
              f"total: {total_correct}/{total_processed} ({total_correct/total_processed*100:.1f}%), "
              f"speed: {rate:.0f} prob/hr, "
              f"ETA: {(len(problems)-batch_start-len(batch))/rate*60:.0f} min")

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Total: {total_correct}/{total_processed} correct ({total_correct/total_processed*100:.1f}%)")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
