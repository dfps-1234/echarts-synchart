# token_compare.py
import os
import random
import config
from utils import load_svg_as_text, load_js_as_text, count_tokens

def get_echarts_js_samples(echarts_dir="/data/home/liyunzhe/echarts_dataset/data/raw/echarts/codes", num_samples=10):
    """从ECharts代码目录随机选取num_samples个JS文件"""
    if not os.path.exists(echarts_dir):
        print(f"ECharts directory not found: {echarts_dir}")
        return []
    js_files = [f for f in os.listdir(echarts_dir) if f.endswith('.js')]
    if len(js_files) == 0:
        return []
    selected = random.sample(js_files, min(num_samples, len(js_files)))
    return [os.path.join(echarts_dir, f) for f in selected]

def get_vgdcu_svg_samples(num_samples=10):
    """从VG-DCU的Plotly和Vega-Lite子集中随机选取num_samples个SVG文件"""
    svg_paths = []
    # Plotly
    plotly_src = config.PLOTLY_SRC
    for ct in os.listdir(plotly_src):
        ct_dir = os.path.join(plotly_src, ct)
        if not os.path.isdir(ct_dir) or ct == 'annotations':
            continue
        svg_files = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.svg')]
        svg_paths.extend(svg_files)
    # Vega-Lite
    vegalite_src = config.VEGALITE_SRC
    subsets = ['bar_vega_dataset', 'line_vega_datasets_3k', 'pie_vega_dataset']
    for sub in subsets:
        sub_dir = os.path.join(vegalite_src, sub)
        if os.path.exists(sub_dir):
            svg_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.svg')]
            svg_paths.extend(svg_files)

    if len(svg_paths) == 0:
        return []
    selected = random.sample(svg_paths, min(num_samples, len(svg_paths)))
    return selected

def compare_tokens():
    """比较SVG和JS的token数量"""
    print("=== Token Comparison ===")
    
    # 获取样本
    js_samples = get_echarts_js_samples(num_samples=config.TOKEN_COMPARE_SAMPLES)
    svg_samples = get_vgdcu_svg_samples(num_samples=config.TOKEN_COMPARE_SAMPLES)

    if not js_samples:
        print("No ECharts JS samples found.")
    if not svg_samples:
        print("No VG-DCU SVG samples found.")

    results = []

    # 计算JS token
    for js_path in js_samples:
        js_text = load_js_as_text(js_path)
        token_count = count_tokens(js_text)
        results.append(('JS', os.path.basename(js_path), token_count))
        print(f"JS: {os.path.basename(js_path)} -> {token_count} tokens")

    # 计算SVG token
    for svg_path in svg_samples:
        svg_text = load_svg_as_text(svg_path)
        token_count = count_tokens(svg_text)
        results.append(('SVG', os.path.basename(svg_path), token_count))
        print(f"SVG: {os.path.basename(svg_path)} -> {token_count} tokens")

    # 简单统计
    js_counts = [r[2] for r in results if r[0] == 'JS']
    svg_counts = [r[2] for r in results if r[0] == 'SVG']
    if js_counts:
        print(f"\nJS average tokens: {sum(js_counts)/len(js_counts):.2f}")
    if svg_counts:
        print(f"SVG average tokens: {sum(svg_counts)/len(svg_counts):.2f}")

    # 如果两者都有，可以比较
    if js_counts and svg_counts:
        ratio = sum(js_counts)/len(js_counts) / (sum(svg_counts)/len(svg_counts))
        print(f"JS/SVG token ratio: {ratio:.2f}")

if __name__ == "__main__":
    compare_tokens()