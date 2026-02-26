"""
Improved GRPO training script (v2)
Key improvements:
- Uses full answer_extraction from utils (handles fractions, LaTeX, etc.)
- Supports num_generations=16 for better advantage estimation
- DAPO loss with proper dynamic sampling
- Better logging and checkpointing
"""
import os
import sys
import re
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["WANDB_PROJECT"] = "mathmodel-reasoning"

from utils.answer_extraction import (
    extract_boxed_answer, extract_answer_from_solution,
    normalize_answer, answers_match
)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Correctness reward using full answer extraction with robust matching"""
    responses = [completion[0]['content'] for completion in completions]

    scores = []
    for resp, true_ans in zip(responses, answer):
        # Multi-strategy extraction
        extracted = extract_answer_from_solution(resp)
        if extracted and answers_match(extracted, true_ans):
            scores.append(2.0)
        else:
            scores.append(0.0)

    # Periodic logging
    if not hasattr(correctness_reward_func, '_call_count'):
        correctness_reward_func._call_count = 0
    correctness_reward_func._call_count += 1

    if correctness_reward_func._call_count % 20 == 1:
        q = prompts[0][-1]['content'][:100]
        extracted_sample = extract_answer_from_solution(responses[0])
        print(f"\n{'='*40}")
        print(f"Q: {q}...")
        print(f"True: {answer[0]}")
        print(f"Pred: {extracted_sample}")
        print(f"Score: {scores[0]}")
        print(f"Batch avg: {sum(scores)/len(scores):.2f}")

    return scores


def format_reward_func(completions, **kwargs) -> list[float]:
    """Format reward for proper boxed answer usage"""
    responses = [completion[0]['content'] for completion in completions]
    scores = []
    for r in responses:
        boxed_count = len(re.findall(r'\\boxed\{', r))
        if boxed_count == 1:
            scores.append(0.5)
        elif boxed_count == 0:
            scores.append(0.0)
        else:
            scores.append(-0.3)
    return scores


def main():
    parser = argparse.ArgumentParser(description='GRPO v2 - Improved training')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data_processing/processed/rl_20k.jsonl')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=5e-7)
    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--num_generations', type=int, default=16)
    parser.add_argument('--generation_batch_size', type=int, default=None,
                        help='vLLM generation batch size (default: same as num_generations)')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_completion_length', type=int, default=4096)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.5)
    parser.add_argument('--vllm_mode', type=str, default='colocate')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--correctness_only', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dapo',
                        choices=['grpo', 'dapo', 'dr_grpo'])
    parser.add_argument('--beta', type=float, default=0.0, help='KL penalty (0=no KL)')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--epsilon_high', type=float, default=None,
                        help='DAPO clip-higher epsilon (None=no clip-higher)')
    parser.add_argument('--mask_truncated', action='store_true',
                        help='Mask truncated completions (DAPO overlong handling)')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = os.path.basename(args.output_dir)

    import wandb
    wandb.login()

    print(f"{'='*60}")
    print(f"GRPO v2 Training Config:")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  LR: {args.learning_rate}, Scheduler: {args.lr_scheduler}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  Generation batch size: {args.generation_batch_size or args.num_generations}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  Epsilon: {args.epsilon}")
    if args.epsilon_high:
        print(f"  Epsilon high (clip-higher): {args.epsilon_high}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Full FT: {trainable/1e6:.0f}M / {total/1e6:.0f}M params")

    # Load data
    samples = []
    with open(args.data_path) as f:
        for line in f:
            samples.append(json.loads(line))
    dataset = Dataset.from_list(samples)
    print(f"Dataset: {len(dataset)} samples")

    # GRPO config
    grpo_kwargs = dict(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        optim="adamw_torch",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size or args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=args.output_dir,
        run_name=args.run_name,
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type=args.loss_type,
        beta=args.beta,
        epsilon=args.epsilon,
    )

    if args.epsilon_high is not None:
        grpo_kwargs['epsilon_high'] = args.epsilon_high

    if args.mask_truncated:
        grpo_kwargs['mask_truncated_completions'] = True

    training_args = GRPOConfig(**grpo_kwargs)

    if args.correctness_only:
        reward_funcs = [correctness_reward_func]
        print("  Reward: correctness only")
    else:
        reward_funcs = [correctness_reward_func, format_reward_func]
        print("  Reward: correctness + format")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nStarting GRPO training...")
    trainer.train()

    # Save final model
    save_dir = os.path.join(args.output_dir, "full_model" if args.lora_rank == 0 else "lora_adapter")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to: {save_dir}")

    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()
