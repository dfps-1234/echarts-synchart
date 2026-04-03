#!/usr/bin/env python3
"""
Chart classification training script with multi-GPU support.
Run with:
    torchrun --nproc_per_node=NUM_GPUs train_chart_classifier.py
or simply:
    python train_chart_classifier.py  (single GPU)
"""

import os
import sys
import gc
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# ========== 环境设置 ==========
# 让 Trainer 自动处理设备分布，不手动设置 CUDA_VISIBLE_DEVICES 或 device_map
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# 可选：关闭 cuDNN 避免某些兼容性问题（如果不必要可以去掉）
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# ========== 配置 ==========
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./output_chart_classification_v3"
TRAIN_JSONL = "/data/home/liyunzhe/echarts_dataset/merged/train.jsonl"
VAL_JSONL = "/data/home/liyunzhe/echarts_dataset/merged/val.jsonl"
IMAGE_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"

BATCH_SIZE = 1          # 每卡 batch size
GRAD_ACCUM = 8          # 梯度累积步数
EPOCHS = 5
LEARNING_RATE = 2e-4

# 显式设置设备为 None，让 Trainer 自动分配
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"可用设备: {device}")

# ========== 处理器 ==========
print("加载处理器...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

# ========== 数据集 ==========
class ChartDataset(Dataset):
    def __init__(self, jsonl_path, image_root):
        self.image_root = Path(image_root)
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        # 过滤无效图片（如果存在）
        self.valid_indices = []
        for idx, sample in enumerate(self.samples):
            img_path = self.image_root / sample["image"]
            if img_path.exists():
                self.valid_indices.append(idx)
        print(f"数据集 {jsonl_path} 共有 {len(self.samples)} 条，有效 {len(self.valid_indices)} 条")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.samples[real_idx]
        img_path = self.image_root / sample["image"]
        conversations = sample["conversations"]
        question = conversations[0]["value"]
        answer = conversations[1]["value"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
        return {"image": image, "question": question, "answer": answer}

# ========== 模型 ==========
print("加载模型...")
# 使用 bfloat16 或 float16
torch_dtype = torch.bfloat16
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch_dtype,
    attn_implementation="eager",  # 可选 "flash_attention_2" 如果安装
    device_map=None,               # 让 Trainer 分配设备
)
model_dtype = next(model.parameters()).dtype

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== Collate 函数（支持多卡） ==========
def make_collate_fn(processor, device):
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        texts = []
        for item in batch:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        images = [item["image"] for item in batch]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # 将张量移动到设备（注意：在 Trainer 中，数据会由 DataLoader 自动分发到设备，这里可先返回CPU）
        # 为了安全，这里只返回 CPU 张量，Trainer 会自动移入 GPU
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.cpu()

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels.cpu()
        return inputs
    return collate_fn

# 不传入 device，collate_fn 返回 CPU 张量，Trainer 会处理到 GPU
collate_fn = make_collate_fn(processor, device)

# ========== 测试前向传播（单卡） ==========
def test_forward(model, dataset, collate_fn):
    if len(dataset) == 0:
        print("数据集为空")
        return False
    sample = dataset[0]
    if sample is None:
        print("样本为 None")
        return False
    batch = collate_fn([sample])
    if batch is None:
        print("collate_fn 返回 None")
        return False
    print("batch keys:", list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    model.eval()
    # 将 batch 移到模型设备（单卡测试用）
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        try:
            outputs = model(**batch)
            print("前向传播成功！")
            return True
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False

print("测试前向传播...")
train_dataset = ChartDataset(TRAIN_JSONL, IMAGE_ROOT)
if not test_forward(model, train_dataset, collate_fn):
    print("前向传播测试失败，退出")
    sys.exit(1)

# 验证集
val_dataset = ChartDataset(VAL_JSONL, IMAGE_ROOT)

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=True if torch_dtype == torch.bfloat16 else False,
    fp16=True if torch_dtype == torch.float16 else False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,           # 多进程数据加载可能冲突，设为0
    ddp_find_unused_parameters=False,   # 加速分布式训练
)

# ========== 训练器 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("开始训练...")
trainer.train()

# 保存
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("训练完成，模型已保存至", OUTPUT_DIR)