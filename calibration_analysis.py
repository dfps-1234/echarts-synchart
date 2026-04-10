#!/usr/bin/env python3
"""
Calibration analysis: compute Expected Calibration Error (ECE) and plot reliability diagram.
Assumes the trained ResNet50 model (no augmentation) and test dataset.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# ---------- Configuration ----------
DATA_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MODEL_PATH = "chart_classifier_resnet50.pth"   # 无增强的最佳模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_BINS = 10          # 等宽分箱数量

# ---------- Load label mapping ----------
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

label_to_id, id_to_label, num_classes = build_label_mapping(TEST_JSONL)

# ---------- Dataset ----------
class ChartDataset:
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset = ChartDataset(TEST_JSONL, IMAGE_DIR, label_to_id, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------- Load model ----------
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------- Collect confidences and correct predictions ----------
all_confidences = []   # 最大 softmax 概率（预测类别的置信度）
all_correct = []       # 是否预测正确（1 或 0）

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)                 # logits
        probs = torch.softmax(outputs, dim=1)   # (B, num_classes)
        max_probs, preds = torch.max(probs, dim=1)
        correct = (preds == labels.to(DEVICE)).float()
        all_confidences.extend(max_probs.cpu().numpy())
        all_correct.extend(correct.cpu().numpy())

all_confidences = np.array(all_confidences)
all_correct = np.array(all_correct)

# ---------- Compute ECE ----------
def compute_ece(confidences, corrects, num_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_conf_in_bin = np.mean(confidences[in_bin])
            accuracy_in_bin = np.mean(corrects[in_bin])
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

ece = compute_ece(all_confidences, all_correct, NUM_BINS)
print(f"Expected Calibration Error (ECE): {ece:.4f}")

# ---------- Plot reliability diagram ----------
fraction_of_positives, mean_predicted_value = calibration_curve(
    all_correct, all_confidences, n_bins=NUM_BINS, strategy='uniform'
)

plt.figure(figsize=(6, 6))
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label="Model")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives (accuracy)")
plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reliability_diagram.png", dpi=300)
print("Reliability diagram saved as reliability_diagram.png")