import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

def is_mostly_blank(image_path, threshold=0.99):
    """判断图像是否几乎纯色（空白图）"""
    try:
        img = Image.open(image_path).convert('RGB')
        data = np.array(img)
        # 如果图像尺寸太小，视为无效
        if data.shape[0] < 10 or data.shape[1] < 10:
            return True
        # 计算唯一颜色数量
        unique_colors = np.unique(data.reshape(-1, 3), axis=0)
        if len(unique_colors) <= 2:
            # 计算主色占比
            colors_flat = data.reshape(-1, 3).dot([1, 256, 256])
            counts = np.bincount(colors_flat)
            main_ratio = counts.max() / counts.sum()
            return main_ratio > threshold
    except Exception as e:
        print(f"Error checking image {image_path}: {e}")
        return True
    return False

def render_one(js_path, png_path, node_script):
    """调用Node脚本渲染单个文件，返回(js_path, success)"""
    cmd = ['node', node_script, str(js_path), str(png_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to render {js_path}: {result.stderr}")
        return js_path, False
    # 检查是否空白图
    if is_mostly_blank(png_path):
        print(f"Blank image detected, deleting {png_path}")
        os.remove(png_path)
        return js_path, False
    return js_path, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='增强后的JS文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='PNG输出目录')
    parser.add_argument('--node_script', type=str, default='./render_echarts.js', help='Node渲染脚本路径')
    parser.add_argument('--num_workers', type=int, default=4, help='并发数')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    js_files = list(input_dir.glob('*.js'))
    print(f"找到 {len(js_files)} 个JS文件，开始渲染...")

    tasks = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for js_path in js_files:
            png_path = output_dir / js_path.with_suffix('.png').name
            tasks.append(executor.submit(render_one, js_path, png_path, args.node_script))

        success_count = 0
        fail_count = 0
        for future in as_completed(tasks):
            js_path, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1

    print(f"渲染完成：成功 {success_count} 个，失败 {fail_count} 个")

if __name__ == '__main__':
    main()