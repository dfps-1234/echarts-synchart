# batch_render_plotly.py
import os
import subprocess
import argparse
from pathlib import Path
from plotly_utils import is_mostly_blank_image
from concurrent.futures import ThreadPoolExecutor, as_completed

def render_one(svg_path, png_path):
    """使用 cairosvg 将 SVG 渲染为 PNG，并检查空白图"""
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    except Exception as e:
        print(f"渲染失败 {svg_path}: {e}")
        return svg_path, False
    if is_mostly_blank_image(png_path):
        print(f"空白图，删除 {png_path}")
        os.remove(png_path)
        return svg_path, False
    return svg_path, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='增强后的SVG目录（包含各子类型）')
    parser.add_argument('--output_dir', type=str, required=True, help='PNG输出根目录')
    parser.add_argument('--workers', type=int, default=4, help='并发数')
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 遍历所有子目录
    svg_files = list(input_root.glob('*/*.svg'))
    print(f"找到 {len(svg_files)} 个SVG文件")

    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for svg_path in svg_files:
            # 保持子目录结构
            rel_path = svg_path.relative_to(input_root)
            png_path = output_root / rel_path.with_suffix('.png')
            png_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(executor.submit(render_one, svg_path, png_path))

        success = 0
        fail = 0
        for future in as_completed(tasks):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"渲染完成：成功 {success} 个，失败 {fail} 个")

if __name__ == '__main__':
    main()