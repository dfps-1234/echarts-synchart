#!/usr/bin/env python3
"""
finetune_on_vgdcu.py
在 VG-DCU 上微调 ECharts-SynChart 预训练的 ResNet50 模型，提升跨数据集泛化性能。
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import glob
import random
import numpy as np

# ==================== 修复 cuDNN 错误 ====================
torch.backends.cudnn.enabled = False

# ==================== 配置 ====================
# 预训练模型路径（你的最佳模型，ResNet50 无数据增强）
PRETRAINED_MODEL = "/data/home/liyunzhe/echarts_dataset/merged/chart_classifier_resnet50.pth"

# VG-DCU 根目录
VGDCU_ROOT = "/data/home/liyunzhe/VG-DCU"
SUBSET = "plotly"  # 使用 plotly 子集

# 类别映射（VG-DCU 目录名 -> 模型标签）
CATEGORY_MAP = {
    "bar": "bar",
    "line": "line",
    "pie": "pie",
    "scatter": "scatter",
    "box": "boxplot",
    "heatmap": "heatmap",
}
# 模型支持的类别列表（与训练时一致）
SUPPORTED_CLASSES = ["bar", "line", "pie", "scatter", "boxplot", "heatmap"]
# 仅使用映射中且在支持列表中的类别
USE_CATEGORIES = [k for k, v in CATEGORY_MAP.items() if v in SUPPORTED_CLASSES]

# 训练超参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出目录
OUTPUT_DIR = "./vgdcu_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图片预处理（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================== 准备 VG-DCU 数据集 ====================
def prepare_vgdcu_dataset(root, subset, categories):
    """扫描 VG-DCU 目录，收集图片路径和标签，返回 list of (img_path, label)"""
    if subset == "plotly":
        base_dir = os.path.join(root, "plotly/user/doushugu/cvl/plotly_group_plain")
    else:
        raise ValueError("Only plotly subset is supported for now")
    
    data = []
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"Warning: {cat_dir} not found, skipping {cat}")
            continue
        png_files = glob.glob(os.path.join(cat_dir, "*.png"))
        label = CATEGORY_MAP[cat]
        for p in png_files:
            data.append((p, label))
    print(f"Total images collected: {len(data)}")
    return data

def split_data(data, test_size=0.2, val_size=0.1):
    """分层划分训练/验证/测试集"""
    # 先分离测试集（20%）
    train_val, test = train_test_split(data, test_size=test_size, stratify=[d[1] for d in data], random_state=42)
    # 再分离验证集（占剩余部分的 1/8，即总体的 10%）
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, stratify=[d[1] for d in train_val], random_state=42)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def save_jsonl(data, filepath):
    """将 (img_path, label) 列表保存为 JSONL 格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for img_path, label in data:
            # 使用相对路径或绝对路径？建议使用绝对路径，方便加载
            item = {
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": "What type of chart is this?"},
                    {"from": "gpt", "value": label}
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==================== 数据集类 ====================
class ChartDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个占位张量，后续 collate 会过滤 None
            return None
        if self.transform:
            image = self.transform(image)
        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

# ==================== 训练函数 ====================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        # 将标签转换为索引（需要 label_to_id 映射）
        label_ids = torch.tensor([label_to_id[l] for l in labels]).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, label_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_ids.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device, label_to_id):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            label_ids = torch.tensor([label_to_id[l] for l in labels]).to(device)
            outputs = model(images)
            loss = criterion(outputs, label_ids)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_ids.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

# ==================== 主程序 ====================
def main():
    print("Using device:", DEVICE)
    
    # 1. 准备 VG-DCU 数据集
    print("Collecting VG-DCU images...")
    data = prepare_vgdcu_dataset(VGDCU_ROOT, SUBSET, USE_CATEGORIES)
    if len(data) == 0:
        print("No data found. Exiting.")
        return
    train_data, val_data, test_data = split_data(data, test_size=0.2, val_size=0.1)
    
    # 保存 JSONL（可选，便于复用）
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "vgdcu_train.jsonl"))
    save_jsonl(val_data, os.path.join(OUTPUT_DIR, "vgdcu_val.jsonl"))
    save_jsonl(test_data, os.path.join(OUTPUT_DIR, "vgdcu_test.jsonl"))
    
    # 构建标签映射（从训练集提取）
    global label_to_id, id_to_label
    all_labels = set([l for _, l in train_data])
    label_list = sorted(all_labels)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    num_classes = len(label_list)
    print(f"Classes in VG-DCU: {label_list}")
    
    # 2. 创建 DataLoader
    train_dataset = ChartDataset(train_data, transform=transform)
    val_dataset = ChartDataset(val_data, transform=transform)
    test_dataset = ChartDataset(test_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # 3. 加载预训练模型并调整分类头
    print("Loading pretrained model...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 注意：原模型是24类，现在可能类别数不同，需调整
    # 加载预训练权重（只加载匹配的部分，fc 不匹配会被忽略）
    pretrained_dict = torch.load(PRETRAINED_MODEL, map_location="cpu")
    model_dict = model.state_dict()
    # 过滤掉 fc 层的权重（因为输出维度不同）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(DEVICE)
    
    # 4. 设置优化器、损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 微调
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE, label_to_id)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ** Best model saved (acc={val_acc:.4f}) **")
    
    # 6. 在测试集上评估最佳模型
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    test_loss, test_acc, preds, refs = evaluate(model, test_loader, criterion, DEVICE, label_to_id)
    print(f"\nTest Results on VG-DCU:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    # 将预测和真实标签转换为类别名
    pred_labels = [id_to_label[p] for p in preds]
    ref_labels = [id_to_label[r] for r in refs]
    print("\nClassification Report:")
    print(classification_report(ref_labels, pred_labels, zero_division=0))
    
    # 保存结果
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(classification_report(ref_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    main()