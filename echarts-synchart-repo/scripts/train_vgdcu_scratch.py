#!/usr/bin/env python3
"""
train_vgdcu_scratch.py
在 VG-DCU 上从头训练 ResNet50（使用 ImageNet 预训练，不使用 ECharts-SynChart 权重）。
用于对比微调实验，证明预训练的价值。
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

# ==================== 修复 cuDNN 错误 ====================
torch.backends.cudnn.enabled = False

# ==================== 配置 ====================
# VG-DCU 根目录
VGDCU_ROOT = "/data/home/liyunzhe/VG-DCU"
SUBSET = "plotly"

# 类别映射（VG-DCU 目录名 -> 标签名，与微调实验保持一致）
CATEGORY_MAP = {
    "bar": "bar",
    "line": "line",
    "pie": "pie",
    "scatter": "scatter",
    "box": "boxplot",
    "heatmap": "heatmap",
}
SUPPORTED_CLASSES = ["bar", "line", "pie", "scatter", "boxplot", "heatmap"]
USE_CATEGORIES = [k for k, v in CATEGORY_MAP.items() if v in SUPPORTED_CLASSES]

# 训练超参数（从头训练使用更高学习率）
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4          # 比微调高，与 ECharts-SynChart 训练一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出目录（与微调区分）
OUTPUT_DIR = "./vgdcu_scratch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================== 准备 VG-DCU 数据集（与微调相同） ====================
def prepare_vgdcu_dataset(root, subset, categories):
    if subset == "plotly":
        base_dir = os.path.join(root, "plotly/user/doushugu/cvl/plotly_group_plain")
    else:
        raise ValueError("Only plotly subset is supported")
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
    train_val, test = train_test_split(data, test_size=test_size,
                                       stratify=[d[1] for d in data], random_state=42)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio,
                                  stratify=[d[1] for d in train_val], random_state=42)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for img_path, label in data:
            item = {
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": "What type of chart is this?"},
                    {"from": "gpt", "value": label}
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==================== 数据集类（与微调相同） ====================
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

# ==================== 训练与评估函数 ====================
def train_one_epoch(model, dataloader, optimizer, criterion, device, label_to_id):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
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

    # 保存 JSONL（可选）
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "vgdcu_train.jsonl"))
    save_jsonl(val_data, os.path.join(OUTPUT_DIR, "vgdcu_val.jsonl"))
    save_jsonl(test_data, os.path.join(OUTPUT_DIR, "vgdcu_test.jsonl"))

    # 构建标签映射
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)

    # 3. 构建模型：使用 ImageNet 预训练，不使用 ECharts-SynChart 权重
    print("Building ResNet50 with ImageNet pretrained weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    # 4. 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. 从头训练
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                criterion, DEVICE, label_to_id)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion,
                                           DEVICE, label_to_id)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ** Best model saved (acc={val_acc:.4f}) **")

    # 6. 测试最佳模型
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    test_loss, test_acc, preds, refs = evaluate(model, test_loader, criterion,
                                                DEVICE, label_to_id)
    print(f"\nTest Results on VG-DCU (trained from scratch):")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    pred_labels = [id_to_label[p] for p in preds]
    ref_labels = [id_to_label[r] for r in refs]
    print("\nClassification Report:")
    print(classification_report(ref_labels, pred_labels, zero_division=0))

    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(classification_report(ref_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    main()