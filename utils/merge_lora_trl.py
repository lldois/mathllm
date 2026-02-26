"""
LoRA权重合并脚本 (TRL版本)
使用peft原生merge，不依赖unsloth
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    max_seq_length: int = 2048,
):
    print(f"加载基座模型: {base_model_path}")
    print(f"加载LoRA适配器: {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, lora_path)

    print("合并LoRA权重...")
    model = model.merge_and_unload()

    print(f"保存到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("合并完成!")


def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重 (TRL版)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    args = parser.parse_args()
    merge_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
