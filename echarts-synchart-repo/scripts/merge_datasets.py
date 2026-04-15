import argparse
import csv
from pathlib import Path
import re

def extract_chart_type_from_js(js_path):
    """从 JS 文件内容中提取图表类型"""
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 匹配 series.type
    match = re.search(r'series:\s*\[\s*{\s*type:\s*[\'"](\w+)[\'"]', content)
    if match:
        return match.group(1)
    # 尝试匹配第二个可能的模式（如 dataset + series）
    match = re.search(r'series:\s*\[[^]]*?type:\s*[\'"](\w+)[\'"]', content, re.DOTALL)
    if match:
        return match.group(1)
    return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--js_dir', required=True, help='原始JS文件目录')
    parser.add_argument('--png_dirs', nargs='+', required=True, help='一个或多个PNG目录（可指定多个）')
    parser.add_argument('--output', default='dataset.csv', help='输出CSV文件名')
    parser.add_argument('--prefix', action='store_true', help='是否在图片路径前添加目录名作为前缀')
    args = parser.parse_args()

    js_dir = Path(args.js_dir)
    # 构建基础名到类型的映射
    type_map = {}
    for js_file in js_dir.glob('*.js'):
        base = js_file.stem
        chart_type = extract_chart_type_from_js(js_file)
        type_map[base] = chart_type
        print(f"{base} -> {chart_type}")

    rows = []
    for png_dir in args.png_dirs:
        png_dir = Path(png_dir)
        for png in png_dir.glob('*.png'):
            # 提取基础名（不含扩展名）
            base = png.stem
            # 对于增强文件，基础名是 example_XX 部分（去掉 _aug_ 后缀）
            if '_aug_' in base:
                base = base.split('_aug_')[0]
            # 检查是否在 type_map 中
            if base not in type_map:
                print(f"警告: {png.name} 找不到对应的类型，跳过")
                continue
            # 图片路径：如果指定了 --prefix，则保存相对路径（含目录名），否则仅文件名
            if args.prefix:
                img_path = f"{png_dir.name}/{png.name}"
            else:
                img_path = png.name
            rows.append({
                'image': img_path,
                'label': type_map[base]
            })

    # 写入CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"合并完成，共 {len(rows)} 条记录，保存至 {args.output}")

if __name__ == '__main__':
    main()