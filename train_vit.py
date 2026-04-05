#!/usr/bin/env python3
"""
train_vit.py - 使用 ViT-Base 进行图表分类
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from PIL import Image
import timm

torch.backends.cudnn.enabled = False

DATA_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"
TRAIN_JSONL = os.path.join(DATA_ROOT, "train.jsonl")
VAL_JSONL = os.path.join(DATA_ROOT, "val.jsonl")
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

BATCH_SIZE = 32  # ViT 需要更小的 batch size
EPOCHS = 15
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = "chart_classifier_vit.pth"

print(f"Using device: {DEVICE}")

def build_label_mapping(jsonl_path):
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
        label_id = self.label_to_id[label]
        return image, label_id

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

def main():
    print("Loading label mapping...")
    label_to_id, id_to_label, num_classes = build_label_mapping(TRAIN_JSONL)
    print(f"Number of classes: {num_classes}")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Building ViT-Base model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  ** Best model saved (acc={val_acc:.4f}) **")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")

    print("Loading best model for testing...")
    model.load_state_dict(torch.load(OUTPUT_MODEL))
    test_loss, test_acc, test_f1, preds, refs = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Weighted F1: {test_f1:.4f}")
    print("\nClassification Report:")
    pred_labels = [id_to_label[p] for p in preds]
    ref_labels = [id_to_label[r] for r in refs]
    print(classification_report(ref_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    main()