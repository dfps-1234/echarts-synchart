import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# 禁用cuDNN
torch.backends.cudnn.enabled = False

# ==================== 配置 ====================
DATA_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"
TRAIN_JSONL = os.path.join(DATA_ROOT, "train.jsonl")
VAL_JSONL = os.path.join(DATA_ROOT, "val.jsonl")
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据比例
DATA_RATIOS = [0.1, 0.25, 0.5, 1.0]

# ==================== 数据集类 ====================
class ChartDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, transform=None, max_samples=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        if max_samples is not None:
            self.samples = random.sample(self.samples, max_samples)
        self.image_dir = image_dir
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
        if not os.path.exists(img_path):
            alt_path = os.path.join(self.image_dir, os.path.basename(sample["image"]))
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = sample["conversations"][1]["value"]
        return image, label

def get_label_to_id(jsonl_path):
    labels = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            labels.add(data["conversations"][1]["value"])
    label_list = sorted(labels)
    return {l: i for i, l in enumerate(label_list)}, label_list

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Training", leave=False):
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
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            label_ids = torch.tensor([label_to_id[l] for l in labels]).to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_ids.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# ==================== 主程序 ====================
print(f"Using device: {DEVICE}")
label_to_id, label_list = get_label_to_id(TRAIN_JSONL)
num_classes = len(label_list)
print(f"Number of classes: {num_classes}")

# 加载完整训练集，获取所有样本
full_dataset = ChartDataset(TRAIN_JSONL, IMAGE_DIR)
total_train_samples = len(full_dataset)
print(f"Total training samples: {total_train_samples}")

results = {}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

for ratio in DATA_RATIOS:
    print(f"\n{'='*50}")
    print(f"Training with {ratio*100:.0f}% of data ({(int(total_train_samples * ratio))} samples)")
    print('='*50)
    
    # 子采样训练集
    n_samples = int(total_train_samples * ratio)
    train_dataset = ChartDataset(TRAIN_JSONL, IMAGE_DIR, transform=transform, max_samples=n_samples)
    val_dataset = ChartDataset(VAL_JSONL, IMAGE_DIR, transform=transform)
    test_dataset = ChartDataset(TEST_JSONL, IMAGE_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        scheduler.step(1 - val_acc)  # 用错误率作为监控指标
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    # 测试最佳模型（这里简单用最后一个epoch，实际可保存最佳）
    test_acc = evaluate(model, test_loader, DEVICE)
    results[ratio] = test_acc
    print(f"Test accuracy for {ratio*100:.0f}% data: {test_acc:.4f}")

print("\n" + "="*50)
print("Data Efficiency Results:")
for ratio, acc in sorted(results.items()):
    print(f"  {ratio*100:.0f}% data: {acc:.4f}")