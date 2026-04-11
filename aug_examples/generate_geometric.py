from PIL import Image
import random

# 打开原始图像
img = Image.open("original.png")

# 1. 随机裁剪（保留 80% 到 95% 的区域）
crop_ratio = random.uniform(0.8, 0.95)
width, height = img.size
new_width = int(width * crop_ratio)
new_height = int(height * crop_ratio)
left = random.randint(0, width - new_width)
top = random.randint(0, height - new_height)
img_cropped = img.crop((left, top, left + new_width, top + new_height))
img_cropped = img_cropped.resize((width, height))  # 缩放回原尺寸，模拟“裁剪后放大”

# 2. 随机旋转（-15 到 15 度）
angle = random.uniform(-15, 15)
img_rotated = img_cropped.rotate(angle, expand=False, fillcolor="white")

# 3. 可选：轻微缩放（模拟几何变换中的尺度变化）
scale = random.uniform(0.9, 1.1)
new_w = int(width * scale)
new_h = int(height * scale)
img_scaled = img_rotated.resize((new_w, new_h))
# 居中放置到原尺寸画布上
final = Image.new("RGB", (width, height), "white")
final.paste(img_scaled, ((width - new_w)//2, (height - new_h)//2))

# 保存结果
final.save("geometric.png")
print("geometric.png 已生成")