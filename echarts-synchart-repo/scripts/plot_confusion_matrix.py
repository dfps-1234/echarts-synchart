#!/usr/bin/env python3
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import transforms, models
from train_classifier import ChartDataset, build_label_mapping  # 复用数据集类

# 配置
DATA_ROOT = "/data/home/liyunzhe/echarts_dataset/merged"
TEST_JSONL = f"{DATA_ROOT}/test.jsonl"
IMAGE_DIR = f"{DATA_ROOT}/images"
MODEL_PATH = "chart_classifier_resnet50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载标签映射
label_to_id, id_to_label, num_classes = build_label_mapping(TEST_JSONL)
print(f"Number of classes: {num_classes}")

# 加载模型
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 数据集和加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_dataset = ChartDataset(TEST_JSONL, IMAGE_DIR, label_to_id, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 收集所有预测和真实标签
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 转换为类别名称
pred_names = [id_to_label[p] for p in all_preds]
true_names = [id_to_label[t] for t in all_labels]

# 计算混淆矩阵
cm = confusion_matrix(true_names, pred_names, labels=list(id_to_label.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id_to_label.values()))

# 绘图（由于类别较多，可以旋转标签或只显示部分）
plt.figure(figsize=(16, 16))
disp.plot(xticks_rotation=90, cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Chart Classification (ResNet50)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("混淆矩阵已保存为 confusion_matrix.png")