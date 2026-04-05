#!/usr/bin/env python3
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms, models

# 禁用 cuDNN 避免初始化错误
torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 加载模型
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 24)
model.load_state_dict(torch.load("chart_classifier_resnet50.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建随机图像（模拟）
dummy_image = Image.new('RGB', (224, 224), color='white')
input_tensor = transform(dummy_image).unsqueeze(0).to(DEVICE)

# 预热
for _ in range(10):
    with torch.no_grad():
        _ = model(input_tensor)

# 测量推理时间
times = []
with torch.no_grad():
    for _ in range(100):
        start = time.time()
        _ = model(input_tensor)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # 毫秒

print(f"Average inference time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")