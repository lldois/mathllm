#!/usr/bin/env python3
"""
Combine multiple SFT datasets into one, with optional upsampling.
Used to merge Math-7B distillation + enhanced distillation + other sources.
"""
import json
import argparse
import random
import os


def load_sft_data(path, upsample=1):
    """Load SFT data, optionally upsampling"""
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    
    if upsample > 1:
        original_len = len(data)
        data = data * upsample
        print(f"  {path}: {original_len} → {len(data)} (×{upsample})")
    else:
        print(f"  {path}: {len(data)} samples")
    
    return data


def ensure_messages_format(item):
    """Ensure item has messages format for SFT training"""
    if 'messages' in item and len(item['messages']) >= 3:
        return item
    
    # Try to reconstruct from text field
    if 'text' in item and item['text']:
        text = item['text']
        # Parse chat template format
        if '<|im_start|>' in text:
            messages = []
            parts = text.split('<|im_start|>')
            for part in parts:
                if not part.strip():
                    continue
                role_end = part.find('\n')
                if role_end == -1:
                    continue
                role = part[:role_end].strip()
                content = part[role_end+1:].replace('<|im_end|>', '').strip()
                if role in ('system', 'user', 'assistant') and content:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 3:
                item['messages'] = messages
                return item
    
    # Fallback: construct from problem/solution fields
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    problem = item.get('problem', '')
    solution = item.get('solution', '')
    if not solution and 'text' in item:
        # Extract assistant response from text
        text = item['text']
        if '<|im_start|>assistant' in text:
            solution = text.split('<|im_start|>assistant\n')[-1].replace('<|im_end|>', '').strip()
    
    if problem and solution:
        item['messages'] = [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
    
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', required=True,
                        help="Input files, optionally with :N for upsampling (e.g., data.jsonl:3)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    all_data = []
    for inp in args.inputs:
        if ':' in inp:
            path, upsample = inp.rsplit(':', 1)
            upsample = int(upsample)
        else:
            path = inp
            upsample = 1
        
        data = load_sft_data(path, upsample)
        for item in data:
            item = ensure_messages_format(item)
            if 'messages' in item and len(item.get('messages', [])) >= 3:
                all_data.append(item)
    
    random.shuffle(all_data)
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nCombined: {len(all_data)} samples → {args.output}")
