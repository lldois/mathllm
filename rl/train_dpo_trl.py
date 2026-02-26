"""
DPO训练脚本 (TRL版本)
使用transformers + peft + trl，不依赖unsloth
适配MetaX C500 (64GB VRAM)
"""
import os
import sys
import json
import argparse
import torch

os.environ["WANDB_PROJECT"] = "mathmodel-reasoning"


def main():
    parser = argparse.ArgumentParser(description="DPO训练 (TRL版)")
    parser.add_argument("--base_model", default="/tmp/pretrainmodel/Qwen2.5-3B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--wandb_run_name", default="dpo_exp")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--rpo_alpha", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    print(f"加载模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if args.lora_rank > 0:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full FT: {trainable/1e6:.0f}M / {total/1e6:.0f}M params")

    # 加载DPO数据
    print(f"加载数据: {args.data_path}")
    data_list = []
    with open(args.data_path) as f:
        for line in f:
            d = json.loads(line)
            prompt_raw = d['prompt']
            chosen_raw = d['chosen']
            rejected_raw = d['rejected']

            if isinstance(prompt_raw, list):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_raw, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt_raw

            if isinstance(chosen_raw, list):
                chosen_text = chosen_raw[0]['content']
            else:
                chosen_text = chosen_raw

            if isinstance(rejected_raw, list):
                rejected_text = rejected_raw[0]['content']
            else:
                rejected_text = rejected_raw

            data_list.append({
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            })

    print(f"加载了 {len(data_list)} 个DPO对")
    dataset = Dataset.from_list(data_list)

    os.makedirs(args.output_dir, exist_ok=True)

    dpo_config = DPOConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        bf16=True,
        output_dir=args.output_dir,
        report_to="wandb",
        run_name=args.wandb_run_name,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        rpo_alpha=args.rpo_alpha,
        label_smoothing=args.label_smoothing,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("开始DPO训练...")
    trainer.train()

    if args.lora_rank > 0:
        save_path = os.path.join(args.output_dir, "lora_adapter")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nLoRA适配器保存到: {save_path}")
    else:
        save_path = os.path.join(args.output_dir, "full_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\n完整模型保存到: {save_path}")

    metrics = trainer.state.log_history
    train_losses = [m['loss'] for m in metrics if 'loss' in m]
    if train_losses:
        print(f"训练完成!")
        print(f"  最终Loss: {train_losses[-1]:.4f}")
        print(f"  平均Loss: {sum(train_losses)/len(train_losses):.4f}")


if __name__ == "__main__":
    main()
