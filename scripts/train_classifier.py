#!/usr/bin/env python3
"""
train_classifier.py
使用 ResNet50 对图表图像进行分类训练
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import argparse
from PIL import Image

# ==================== 修复 cuDNN 错误 ====================
torch.backends.cudnn.enabled = False   # 禁用 cuDNN，避免初始化错误
# =======================================================

# ==================== 配置 ====================
DATA_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"
TRAIN_JSONL = os.path.join(DATA_ROOT, "train.jsonl")
VAL_JSONL = os.path.join(DATA_ROOT, "val.jsonl")
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")   # 图片在 merged/images/ 下

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = "chart_classifier_resnet50.pth"

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    # 如果 cuDNN 启用但出错，我们已禁用，所以这里会显示 False

# ==================== 辅助函数 ====================
def build_label_mapping(jsonl_path):
    """从训练集 JSONL 中构建 label -> id 映射"""
    labels = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            label = data["conversations"][1]["value"]
            labels.add(label)
    label_list = sorted(labels)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    return label_to_id, id_to_label, len(label_list)

# ==================== 数据集 ====================
class ChartDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, label_to_id, transform=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.image_dir = image_dir
        self.label_to_id = label_to_id
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        # 如果图片路径不包含 images/ 前缀，则直接拼接
        if not os.path.exists(img_path):
            # 尝试去掉可能重复的 images/ 前缀
            alt_path = os.path.join(self.image_dir, os.path.basename(sample["image"]))
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = sample["conversations"][1]["value"]
        label_id = self.label_to_id[label]
        return image, label_id

# ==================== 训练函数 ====================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1, all_preds, all_labels

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--output', type=str, default=OUTPUT_MODEL)
    args = parser.parse_args()

    print("Loading label mapping from training set...")
    label_to_id, id_to_label, num_classes = build_label_mapping(TRAIN_JSONL)
    print(f"Number of classes: {num_classes}")
    print("Classes:", list(label_to_id.keys())[:10], "...")

    # 数据增强（训练集）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = ChartDataset(TRAIN_JSONL, IMAGE_DIR, label_to_id, transform=train_transform)
    val_dataset = ChartDataset(VAL_JSONL, IMAGE_DIR, label_to_id, transform=val_transform)
    test_dataset = ChartDataset(TEST_JSONL, IMAGE_DIR, label_to_id, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Building model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  ** Best model saved (acc={val_acc:.4f}) **")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")

    # 测试最佳模型
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(args.output))
    test_loss, test_acc, test_f1, preds, refs = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Weighted F1: {test_f1:.4f}")
    print("\nClassification Report:")
    # 将预测和真实标签转换为原始类别名
    pred_labels = [id_to_label[p] for p in preds]
    ref_labels = [id_to_label[r] for r in refs]
    print(classification_report(ref_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    main()