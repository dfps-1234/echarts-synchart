import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def is_mostly_blank(image_path, threshold=0.99):
    try:
        img = Image.open(image_path).convert('RGB')
        data = np.array(img)
        if data.shape[0] < 10 or data.shape[1] < 10:
            return True
        unique_colors = np.unique(data.reshape(-1, 3), axis=0)
        if len(unique_colors) <= 2:
            colors_flat = data.reshape(-1, 3).dot([1, 256, 256])
            counts = np.bincount(colors_flat)
            main_ratio = counts.max() / counts.sum()
            return main_ratio > threshold
    except Exception:
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='检测并可选删除空白PNG')
    parser.add_argument('--png_dir', required=True, help='PNG图片目录')
    parser.add_argument('--threshold', type=float, default=0.99, help='空白判定阈值（主色占比）')
    parser.add_argument('--delete', action='store_true', help='删除检测到的空白图')
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    png_files = list(png_dir.glob('*.png'))
    blank = []
    for png in png_files:
        if is_mostly_blank(png, args.threshold):
            blank.append(png)
    print(f"总 PNG 数量: {len(png_files)}")
    print(f"空白图数量: {len(blank)}")
    if blank:
        print("空白图列表 (前20个):")
        for f in blank[:20]:
            print(f"  {f.name}")
        if args.delete:
            print("正在删除空白图...")
            for f in blank:
                f.unlink()
            print("删除完成。")
    else:
        print("未发现空白图。")

if __name__ == '__main__':
    main()