"""
Fast SFT training using pre-tokenized data + transformers Trainer.
Avoids TRL's slow preprocessing pipeline.
Supports full fine-tuning (lora_rank=0) and LoRA.
"""
import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


class PreTokenizedDataset(TorchDataset):
    """Pre-tokenize all data at init for fast training start."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        
        print(f"Loading and tokenizing data from {data_path}...")
        raw_data = []
        with open(data_path, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line))
        
        print(f"Tokenizing {len(raw_data)} samples...")
        skipped = 0
        for i, record in enumerate(raw_data):
            if i % 5000 == 0 and i > 0:
                print(f"  Tokenized {i}/{len(raw_data)}...")
            
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
            elif "text" in record:
                text = record["text"]
            else:
                skipped += 1
                continue
            
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            
            encoded = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length, 
                padding=False,
                return_tensors=None
            )
            
            if len(encoded["input_ids"]) < 10:
                skipped += 1
                continue
            
            self.items.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"].copy(),
            })
        
        print(f"Tokenized {len(self.items)} samples (skipped {skipped})")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class PaddingCollator:
    """Dynamic padding collator - pads to longest in batch."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="Fast SFT Training")
    parser.add_argument("--base_model", type=str, default="/tmp/pretrainmodel/Qwen2.5-3B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="mathmodel-reasoning")
    parser.add_argument("--wandb_run_name", type=str, default="sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    print(f"{'='*60}")
    print(f"Fast SFT Training")
    print(f"Model: {args.base_model}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank} ({'Full FT' if args.lora_rank == 0 else 'LoRA'})")
    print(f"LR: {args.learning_rate}, Epochs: {args.num_epochs}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-tokenize dataset
    dataset = PreTokenizedDataset(args.data_path, tokenizer, args.max_seq_length)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full FT: {total:,} parameters")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        report_to="wandb",
        run_name=args.wandb_run_name,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    
    # Trainer
    collator = PaddingCollator(tokenizer, args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    
    print(f"Training samples: {len(dataset)}")
    total_steps = len(dataset) // (args.batch_size * args.gradient_accumulation_steps)
    print(f"Estimated steps/epoch: {total_steps}")
    print(f"Starting training...")
    
    stats = trainer.train()
    
    # Save
    save_path = os.path.join(args.output_dir, "final_model")
    if args.lora_rank > 0:
        model.save_pretrained(save_path)
    else:
        trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    info = {
        "base_model": args.base_model,
        "data_path": args.data_path,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "train_samples": len(dataset),
        "train_runtime": stats.metrics.get("train_runtime", 0),
        "train_loss": stats.metrics.get("train_loss", 0),
    }
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Runtime: {stats.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Final loss: {stats.metrics.get('train_loss', 0):.4f}")
    print(f"  Model saved to: {save_path}")


if __name__ == "__main__":
    main()
