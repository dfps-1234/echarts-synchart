#!/usr/bin/env python3
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./output_chart_classification_v3', help='训练好的模型路径')
    parser.add_argument('--test_jsonl', default='test.jsonl', help='测试集 JSONL 文件')
    parser.add_argument('--image_root', default='/data/home/liyunzhe/echarts_dataset/merged', help='图片根目录')
    parser.add_argument('--base_model', default='Qwen/Qwen2-VL-2B-Instruct', help='基座模型名称')
    parser.add_argument('--batch_size', type=int, default=1, help='评估时 batch size')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用 CPU（会非常慢）')
    return parser.parse_args()

def load_model(model_path, base_model, use_cpu=False):
    processor = AutoProcessor.from_pretrained(base_model, use_fast=False)
    device = torch.device("cpu" if use_cpu else "cuda:0")
    dtype = torch.float32 if use_cpu else torch.bfloat16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device} if not use_cpu else None,
    )
    if not use_cpu:
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # CPU 上加载 LoRA 需要先加载到 CPU
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    # 清理缓存
    if not use_cpu:
        torch.cuda.empty_cache()
    gc.collect()
    return processor, model

def evaluate(model, processor, test_jsonl, image_root, use_cpu=False):
    with open(test_jsonl, 'r') as f:
        samples = [json.loads(line) for line in f]
    predictions = []
    references = []
    for sample in tqdm(samples, desc="Evaluating"):
        img_path = Path(image_root) / sample["image"]
        if not img_path.exists():
            print(f"图片不存在: {img_path}")
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
            continue
        question = sample["conversations"][0]["value"]
        true_label = sample["conversations"][1]["value"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        # 移动到设备
        if not use_cpu:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        if "Assistant:" in generated_text:
            pred = generated_text.split("Assistant:")[-1].strip()
        else:
            pred = generated_text.strip()
        predictions.append(pred)
        references.append(true_label)

    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    print(f"\n准确率 (Accuracy): {acc:.4f}")
    print(f"加权 F1 分数: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(references, predictions))
    return acc, f1

if __name__ == "__main__":
    args = parse_args()
    processor, model = load_model(args.model_path, args.base_model, args.use_cpu)
    evaluate(model, processor, args.test_jsonl, args.image_root, args.use_cpu)