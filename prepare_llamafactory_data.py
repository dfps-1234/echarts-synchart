#!/usr/bin/env python3
import os
import json
import random
from pathlib import Path

# 路径配置
IMG_DIR = "images"
CODE_DIR = "codes"
OUTPUT_DIR = "../../LLaMA-Factory/data/chart_js"  # 放在 LLaMA-Factory 的 data 下
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 收集所有图片和代码对
pairs = []
for img_file in os.listdir(IMG_DIR):
    if not img_file.endswith(".png"):
        continue
    base = img_file[:-4]
    js_file = os.path.join(CODE_DIR, base + ".js")
    if os.path.exists(js_file):
        with open(js_file, "r", encoding="utf-8") as f:
            js_code = f.read()
        pairs.append({
            "image": os.path.join(IMG_DIR, img_file),
            "js": js_code
        })

# 随机打乱并划分训练/验证/测试 (80/10/10)
random.shuffle(pairs)
total = len(pairs)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]

# 定义转换为 LLaMA-Factory 对话格式的函数
def convert_to_sharegpt(pairs, split_name):
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            entry = {
                "image": p["image"],
                "conversations": [
                    {"from": "human", "value": "Generate JavaScript code for this chart."},
                    {"from": "gpt", "value": p["js"]}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} samples to {output_file}")

convert_to_sharegpt(train_pairs, "train")
convert_to_sharegpt(val_pairs, "val")
convert_to_sharegpt(test_pairs, "test")

print(f"Total pairs: {total}, train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")