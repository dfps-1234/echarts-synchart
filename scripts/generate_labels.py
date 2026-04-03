import os
import re
import csv
from pathlib import Path
import argparse
import json

def extract_chart_type(js_content):
    """从 JS 内容中提取图表类型（简单规则）"""
    # 尝试匹配 option.series[0].type
    match = re.search(r'series:\s*\[\s*{\s*type:\s*[\'"](\w+)[\'"]', js_content)
    if match:
        return match.group(1)
    # 如果没找到，尝试匹配 series 数组中多个类型（取第一个）
    match = re.search(r'series:\s*\[([^]]*?)type:\s*[\'"](\w+)[\'"]', js_content, re.DOTALL)
    if match:
        return match.group(2)
    # 尝试匹配 dataset + 多个 series（如 example_85.js，但这类很少）
    return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--js_dir', required=True, help='原始 ECharts JS 文件目录')
    parser.add_argument('--png_dir', required=True, help='渲染后的 PNG 目录')
    parser.add_argument('--output', default='labels.csv', help='输出 CSV 文件路径')
    parser.add_argument('--aug_suffix', default='_aug_', help='增强文件名的后缀标识')
    args = parser.parse_args()

    js_dir = Path(args.js_dir)
    png_dir = Path(args.png_dir)
    output_path = Path(args.output)

    # 先读取原始 JS，建立基础名 -> 图表类型映射
    type_map = {}
    for js_file in js_dir.glob('*.js'):
        base = js_file.stem
        with open(js_file, 'r', encoding='utf-8') as f:
            content = f.read()
        chart_type = extract_chart_type(content)
        type_map[base] = chart_type
        print(f"{base} -> {chart_type}")

    # 遍历 PNG 文件，生成标签
    rows = []
    for png in png_dir.glob('*.png'):
        # 文件名格式：example_X_aug_YYY.png
        parts = png.stem.split('_aug_')
        if len(parts) != 2:
            continue
        base_name = parts[0]  # 例如 example_11
        if base_name not in type_map:
            print(f"Warning: No type for {base_name}")
            continue
        rows.append({
            'image': png.name,
            'label': type_map[base_name]
        })

    # 写入 CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"生成标签文件 {output_path}，共 {len(rows)} 条记录。")

if __name__ == '__main__':
    main()