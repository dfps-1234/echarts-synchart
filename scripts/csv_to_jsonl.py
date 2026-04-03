#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path

def convert(csv_path, output_path, question="What type of chart is this?"):
    with open(csv_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for row in reader:
                # 图片路径可能已经包含 "images/" 前缀
                img_path = row['image']
                label = row['label']
                item = {
                    "image": img_path,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": label}
                    ]
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
    print(f"转换完成，共 {count} 条记录，保存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='输入的 CSV 文件路径')
    parser.add_argument('--output', required=True, help='输出的 JSONL 文件路径')
    parser.add_argument('--question', default='What type of chart is this?',
                        help='向模型提出的问题')
    args = parser.parse_args()
    convert(args.csv, args.output, args.question)