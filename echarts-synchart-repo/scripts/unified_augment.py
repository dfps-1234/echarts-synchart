"""
统一增强脚本：整合 augment_echarts.py 和 custom_augment.py
用法：
    python unified_augment.py --input_dir <原始JS目录> --output_dir <输出目录> --num 5 --workers 8
"""

import os
import re
import random
import json
import argparse
import ast
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import VALUE_RELATIVE_RANGE, VALUE_ABSOLUTE_RANGE, TEXT_LABELS_POOL
from utils import set_random_seed, random_color_variation, perturb_number

# ==================== 通用增强函数（来自 augment_echarts.py） ====================

def find_matching_bracket(text, start_pos):
    """找到从 start_pos 开始的匹配括号的结束位置（处理嵌套）"""
    stack = []
    for i in range(start_pos, len(text)):
        if text[i] == '[':
            stack.append(i)
        elif text[i] == ']':
            stack.pop()
            if not stack:
                return i
    return -1

def extract_full_array(content, start_idx):
    """从 start_idx 开始提取完整的数组字符串（包括嵌套）"""
    if content[start_idx] != '[':
        return None
    end_idx = find_matching_bracket(content, start_idx)
    if end_idx == -1:
        return None
    return content[start_idx:end_idx+1]

def replace_numbers_in_array(array_str):
    """处理数组字符串中的数字（支持嵌套），返回修改后的数组字符串。"""
    def process_item(item):
        if isinstance(item, (int, float)):
            return perturb_number(item, VALUE_RELATIVE_RANGE, VALUE_ABSOLUTE_RANGE)
        elif isinstance(item, list):
            return [process_item(sub) for sub in item]
        elif isinstance(item, dict):
            new_dict = {}
            for k, v in item.items():
                if k in ['value', 'y', 'x', 'data'] and isinstance(v, (int, float)):
                    new_dict[k] = perturb_number(v, VALUE_RELATIVE_RANGE, VALUE_ABSOLUTE_RANGE)
                elif isinstance(v, (list, dict)):
                    new_dict[k] = process_item(v)
                else:
                    new_dict[k] = v
            return new_dict
        else:
            return item
    try:
        cleaned = array_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        obj = ast.literal_eval(cleaned)
        new_obj = process_item(obj)
        return json.dumps(new_obj, ensure_ascii=False)
    except Exception:
        # 降级：直接正则替换数字
        def replace_num(m):
            num = m.group(0)
            try:
                n = float(num)
                new_n = perturb_number(n, VALUE_RELATIVE_RANGE, VALUE_ABSOLUTE_RANGE)
                return str(int(new_n)) if '.' not in num else f"{new_n:.2f}"
            except:
                return num
        return re.sub(r'\b\d+\.?\d*\b', replace_num, array_str)

def replace_colors(match):
    """替换颜色值，支持十六进制和 rgb/rgba 中的数字"""
    if match.group(1):  # 无引号十六进制
        color = match.group(1)
        new_color = random_color_variation('#' + color)
        return new_color
    elif match.group(2):  # 单引号十六进制
        color = match.group(2)
        new_color = random_color_variation('#' + color)
        return f"'{new_color}'"
    elif match.group(3):  # 双引号十六进制
        color = match.group(3)
        new_color = random_color_variation('#' + color)
        return f'"{new_color}"'
    elif match.group(4):  # rgb/rgba
        func = match.group(4)
        nums_str = match.group(5)
        nums = re.findall(r'\d+\.?\d*', nums_str)
        if len(nums) >= 3:
            r = int(perturb_number(float(nums[0]), 0.2, 20))
            g = int(perturb_number(float(nums[1]), 0.2, 20))
            b = int(perturb_number(float(nums[2]), 0.2, 20))
            if func == 'rgb':
                return f"rgb({r},{g},{b})"
            else:  # rgba
                a = float(nums[3]) if len(nums) > 3 else 1.0
                a = max(0, min(1, a + random.uniform(-0.2, 0.2)))
                return f"rgba({r},{g},{b},{a:.2f})"
    return match.group(0)

def replace_labels(array_str):
    """替换全字符串数组的标签，如 '['Mon','Tue',...]' 返回新字符串"""
    strings = re.findall(r"'([^']*)'|\"([^\"]*)\"", array_str)
    strings = [s[0] if s[0] else s[1] for s in strings]
    if not strings:
        return array_str
    pool = random.choice(TEXT_LABELS_POOL)
    if len(pool) < len(strings):
        new_strings = random.choices(pool, k=len(strings))
    else:
        new_strings = random.sample(pool, len(strings))
    new_str = "[" + ", ".join([f"'{s}'" for s in new_strings]) + "]"
    return new_str

def general_augment(content, num_variations, seed):
    """通用增强逻辑：对 content 进行 num_variations 次修改，返回生成的内容列表"""
    results = []
    random.seed(seed)
    for i in range(num_variations):
        new_content = content
        # 查找所有 data: 的位置
        data_positions = [m.start() for m in re.finditer(r'data\s*:', new_content)]
        # 从后往前替换，避免位置变化
        for pos in reversed(data_positions):
            after_data = new_content[pos+5:]
            match = re.search(r'\[\s*', after_data)
            if not match:
                continue
            start_bracket = pos + 5 + match.start()
            array_str = extract_full_array(new_content, start_bracket)
            if array_str:
                new_array = replace_numbers_in_array(array_str)
                new_content = new_content[:start_bracket] + new_array + new_content[start_bracket+len(array_str):]

        # 替换颜色
        new_content = re.sub(
            r'#([0-9A-Fa-f]{6})\b|(?<=\')([0-9A-Fa-f]{6})(?=\')|(?<=")([0-9A-Fa-f]{6})(?=")|(rgb|rgba)\(([^)]+)\)',
            replace_colors,
            new_content
        )

        # 替换全字符串数组的标签
        for pos in reversed(data_positions):
            after_data = new_content[pos+5:]
            match = re.search(r'\[\s*', after_data)
            if not match:
                continue
            start_bracket = pos + 5 + match.start()
            array_str = extract_full_array(new_content, start_bracket)
            if array_str and not re.search(r'\d', array_str):
                new_array = replace_labels(array_str)
                new_content = new_content[:start_bracket] + new_array + new_content[start_bracket+len(array_str):]

        results.append(new_content)
    return results

# ==================== 定制修改器（来自 custom_augment.py） ====================

# 文件名到修改函数的映射
MODIFIERS = {}

def modifier_for(filename):
    def decorator(func):
        MODIFIERS[filename] = func
        return func
    return decorator

# ---------- 辅助函数（定制修改器内部使用） ----------
def _replace_numbers_in_str(s, relative_range=0.3, absolute_range=10):
    """替换字符串中的所有数字，保留格式"""
    def repl(m):
        num_str = m.group(0)
        try:
            num = float(num_str)
            new_num = perturb_number(num, relative_range, absolute_range)
            if '.' in num_str:
                return f"{new_num:.2f}"
            else:
                return str(int(new_num))
        except:
            return num_str
    return re.sub(r'\b\d+\.?\d*\b', repl, s)

def _replace_colors_in_str(s):
    """替换十六进制和rgb/rgba颜色"""
    def repl_color(m):
        if m.group(1):  # 无引号十六进制
            return random_color_variation('#' + m.group(1))
        elif m.group(2):  # 单引号十六进制
            return "'" + random_color_variation('#' + m.group(2)) + "'"
        elif m.group(3):  # 双引号十六进制
            return '"' + random_color_variation('#' + m.group(3)) + '"'
        elif m.group(4):  # rgb/rgba
            nums = re.findall(r'\d+\.?\d*', m.group(5))
            if len(nums) >= 3:
                r = int(perturb_number(float(nums[0]), 0.2, 20))
                g = int(perturb_number(float(nums[1]), 0.2, 20))
                b = int(perturb_number(float(nums[2]), 0.2, 20))
                if m.group(4) == 'rgb':
                    return f"rgb({r},{g},{b})"
                else:
                    a = float(nums[3]) if len(nums) > 3 else 1.0
                    a = max(0, min(1, a + random.uniform(-0.2, 0.2)))
                    return f"rgba({r},{g},{b},{a:.2f})"
        return m.group(0)
    return re.sub(r'#([0-9A-Fa-f]{6})\b|(?<=\')([0-9A-Fa-f]{6})(?=\')|(?<=")([0-9A-Fa-f]{6})(?=")|(rgb|rgba)\(([^)]+)\)', repl_color, s)

# ---------- 修改器定义 ----------
@modifier_for('example_8.js')
def modify_8(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_10.js')
def modify_10(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_11.js')
def modify_11(content):
    # 只修改数据数组中的数值，不修改 visualMap 的 min/max
    # 使用正则匹配 series 中的 data 数组
    def repl_series_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return f'data: {new_data}'
    # 匹配 series 中的 data: [...]
    content = re.sub(r'data:\s*(\[.*?\])', repl_series_data, content, flags=re.DOTALL)
    # 同时修改颜色（但注意 visualMap 中的颜色可能也被修改，但问题不大）
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_14.js')
def modify_14(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_15.js')
def modify_15(content):
    # 只修改 series 中的 data 数组（两处）
    # 使用正则匹配 series 中 data 数组，并替换数值
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return f'data: {new_data}'
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    # 修改颜色
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_16.js')
def modify_16(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_21.js')
def modify_21(content):
    def repl_func(m):
        func_body = m.group(1)
        new_body = _replace_numbers_in_str(func_body, 0.3, 2)
        return f'function func(x) {{ {new_body} }}'
    content = re.sub(r'function\s+func\s*\(\s*x\s*\)\s*\{([^}]+)\}', repl_func, content, flags=re.DOTALL)
    return content

@modifier_for('example_22.js')
def modify_22(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_28.js')
def modify_28(content):
    # 不修改 N_POINT，保持原值
    # 只修改颜色和数据
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_30.js')
def modify_30(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'data: ' + new_data
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_31.js')
def modify_31(content):
    return modify_30(content)

@modifier_for('example_32.js')
def modify_32(content):
    def repl_data_def(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data_def, content, flags=re.DOTALL)
    return content

@modifier_for('example_33.js')
def modify_33(content):
    def repl_r(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.3, 2)
        return f'let r = {new_expr};'
    content = re.sub(r'let\s+r\s*=\s*([^;]+);', repl_r, content)
    return content

@modifier_for('example_34.js')
def modify_34(content):
    return modify_33(content)

@modifier_for('example_36.js')
def modify_36(content):
    return modify_32(content)

@modifier_for('example_37.js')
def modify_37(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_39.js')
def modify_39(content):
    def repl_source(m):
        source_str = m.group(1)
        new_source = _replace_numbers_in_str(source_str, 0.3, 5)
        return 'source: ' + new_source
    content = re.sub(r'source:\s*(\[.*?\])', repl_source, content, flags=re.DOTALL)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_51.js')
def modify_51(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'data: ' + new_data
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    content = _replace_numbers_in_str(content, 0.3, 1000)
    return content

@modifier_for('example_52.js')
def modify_52(content):
    # 只修改 data 数组中的数值，保留 yMax 不变
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return f'data: {new_data}'
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    # 修改颜色
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_53.js')
def modify_53(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_54.js')
def modify_54(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_57.js')
def modify_57(content):
    def repl_rawdata(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 50)
        return 'const rawData = ' + new_data
    content = re.sub(r'const\s+rawData\s*=\s*(\[.*?\]);', repl_rawdata, content, flags=re.DOTALL)
    def repl_color_array(m):
        color_str = m.group(1)
        new_color_str = _replace_colors_in_str(color_str)
        return 'const color = ' + new_color_str
    content = re.sub(r'const\s+color\s*=\s*(\[.*?\]);', repl_color_array, content, flags=re.DOTALL)
    return content

@modifier_for('example_62.js')
def modify_62(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_64.js')
def modify_64(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_66.js')
def modify_66(content):
    def repl_gen_expr(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.3, 2)
        return f'data1.push(({new_expr}));'
    content = re.sub(r'data1\.push\(\((.*?)\)\);', repl_gen_expr, content)
    def repl_gen_expr2(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.3, 2)
        return f'data2.push(({new_expr}));'
    content = re.sub(r'data2\.push\(\((.*?)\)\);', repl_gen_expr2, content)
    content = re.sub(r'for\s*\(var\s+i\s*=\s*0;\s*i\s*<\s*(\d+);\s*i\+\+\)',
                     lambda m: f'for (var i = 0; i < {random.randint(50,150)}; i++)', content)
    return content

@modifier_for('example_68.js')
def modify_68(content):
    # 修改 dataCount 为一个合理范围内的随机数
    def repl_datacount(m):
        # 原始值可能是 5e5 或 500000
        new_count = random.randint(1000, 5000)
        return str(new_count)
    content = re.sub(r'dataCount\s*=\s*(\d+\.?\d*e?\d*)', repl_datacount, content)
    # 修改 generateData 函数体中的数字（循环内的表达式等）
    def repl_func(m):
        func_body = m.group(1)
        new_body = _replace_numbers_in_str(func_body, 0.3, 20)
        return f'function generateData(count) {{{new_body}}}'
    content = re.sub(r'function\s+generateData\s*\(\s*count\s*\)\s*\{([^}]+)\}', repl_func, content, flags=re.DOTALL)
    # 修改颜色
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_70.js')
def modify_70(content):
    def repl_data_item(m):
        prefix = m.group(1)
        num_str = m.group(2)
        suffix = m.group(3)
        new_num = perturb_number(float(num_str), 0.3, 2)
        return f"{prefix}{int(new_num)}{suffix}"
    content = re.sub(r"(\['[^']*',\s*)(\d+)(\s*,.*?\])", repl_data_item, content)
    return content

@modifier_for('example_71.js')
def modify_71(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_72.js')
def modify_72(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_74.js')
def modify_74(content):
    # 只修改颜色，不修改任何数值
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_75.js')
def modify_75(content):
    # 只修改颜色，不修改数值
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_80.js')
def modify_80(content):
    def repl_breaks(m):
        breaks_str = m.group(1)
        new_breaks = _replace_numbers_in_str(breaks_str, 0.3, 500)
        return 'var _currentAxisBreaks = ' + new_breaks
    content = re.sub(r'var\s+_currentAxisBreaks\s*=\s*(\[.*?\]);', repl_breaks, content, flags=re.DOTALL)
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 50)
        return 'data: ' + new_data
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    return content

@modifier_for('example_81.js')
def modify_81(content):
    def repl_source(m):
        source_str = m.group(1)
        new_source = _replace_numbers_in_str(source_str, 0.3, 10)
        return 'source: ' + new_source
    content = re.sub(r'source:\s*(\[.*?\])', repl_source, content, flags=re.DOTALL)
    return content

@modifier_for('example_83.js')
def modify_83(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_85.js')
def modify_85(content):
    # 可以安全修改数值和颜色
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_86.js')
def modify_86(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_87.js')
def modify_87(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 20)
        return 'data: ' + new_data
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_88.js')
def modify_88(content):
    return modify_87(content)

@modifier_for('example_89.js')
def modify_89(content):
    return modify_87(content)

@modifier_for('example_90.js')
def modify_90(content):
    return modify_87(content)

@modifier_for('example_91.js')
def modify_91(content):
    return modify_87(content)

@modifier_for('example_96.js')
def modify_96(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 5)
        return f'value: {new_num}'
    content = re.sub(r'value:\s*(\d+\.?\d*)', repl_value, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    return content

@modifier_for('example_97.js')
def modify_97(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 2)
        return f'value: {new_num}'
    content = re.sub(r'value:\s*(\d+\.?\d*)', repl_value, content)
    return content

@modifier_for('example_101.js')
def modify_101(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_102.js')
def modify_102(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_103.js')
def modify_103(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_105.js')
def modify_105(content):
    def repl_source(m):
        source_str = m.group(1)
        new_source = _replace_numbers_in_str(source_str, 0.3, 5)
        return 'source: ' + new_source
    content = re.sub(r'source:\s*(\[.*?\])', repl_source, content, flags=re.DOTALL)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_114.js')
def modify_114(content):
    def repl_y(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.3, 2)
        return f'const y = {new_expr};'
    content = re.sub(r'const\s+y\s*=\s*([^;]+);', repl_y, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_116.js')
def modify_116(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data, content, flags=re.DOTALL)
    return content

@modifier_for('example_118.js')
def modify_118(content):
    def repl_female(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const femaleData = ' + new_data
    content = re.sub(r'const\s+femaleData\s*=\s*(\[.*?\]);', repl_female, content, flags=re.DOTALL)
    def repl_male(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const maleDeta = ' + new_data
    content = re.sub(r'const\s+maleDeta\s*=\s*(\[.*?\]);', repl_male, content, flags=re.DOTALL)
    return content

@modifier_for('example_119.js')
def modify_119(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data, content, flags=re.DOTALL)
    return content

@modifier_for('example_120.js')
def modify_120(content):
    return modify_119(content)

@modifier_for('example_134.js')
def modify_134(content):
    def repl_rawdata(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return 'const rawData = ' + new_data
    content = re.sub(r'const\s+rawData\s*=\s*(\[.*?\]);', repl_rawdata, content, flags=re.DOTALL)
    def repl_colors(m):
        colors_str = m.group(1)
        new_colors = _replace_colors_in_str(colors_str)
        return 'color: ' + new_colors
    content = re.sub(r'color:\s*(\[.*?\])', repl_colors, content, flags=re.DOTALL)
    return content

@modifier_for('example_135.js')
def modify_135(content):
    def repl_random(m):
        return 'Math.random() * ' + str(random.randint(500, 1500))
    content = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', repl_random, content)
    def repl_graph(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'[{num_str}, {new_num}]'
    content = re.sub(r'\[\s*\'[^\']+\'\s*,\s*(\d+)\s*\]', repl_graph, content)
    content = re.sub(r'min:\s*(\d+)', lambda m: f'min: {random.randint(0, 200)}', content)
    content = re.sub(r'max:\s*(\d+)', lambda m: f'max: {random.randint(800, 1200)}', content)
    return content

@modifier_for('example_160.js')
def modify_160(content):
    def repl_rawdata(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 50)
        return 'const rawData = ' + new_data
    content = re.sub(r'const\s+rawData\s*=\s*(\[.*?\])\s*\.reverse\(\)', repl_rawdata, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_165.js')
def modify_165(content):
    def repl_value(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.3, 5)
        return f'value: [{new_expr}]'
    content = re.sub(r'value:\s*\[\s*(.*?)\s*\]', repl_value, content, flags=re.DOTALL)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_168.js')
def modify_168(content):
    def repl_source(m):
        source_str = m.group(1)
        new_source = _replace_numbers_in_str(source_str, 0.3, 20)
        return 'source: ' + new_source
    content = re.sub(r'source:\s*(\[.*?\])', repl_source, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_169.js')
def modify_169(content):
    return modify_168(content)

@modifier_for('example_170.js')
def modify_170(content):
    def repl_random(m):
        return 'Math.random() * ' + str(random.randint(100, 300))
    content = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', repl_random, content)
    content = re.sub(r'min:\s*(-?\d+)', lambda m: f'min: {random.randint(-500, -300)}', content)
    content = re.sub(r'max:\s*(\d+)', lambda m: f'max: {random.randint(500, 700)}', content)
    content = re.sub(r'start:\s*(\d+)', lambda m: f'start: {random.randint(0, 10)}', content)
    content = re.sub(r'end:\s*(\d+)', lambda m: f'end: {random.randint(20, 40)}', content)
    return content

@modifier_for('example_171.js')
def modify_171(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\])\s*\.map', repl_data, content, flags=re.DOTALL)
    content = re.sub(r'min:\s*(\d+)', lambda m: f'min: {random.randint(0, 5)}', content)
    content = re.sub(r'max:\s*(\d+)', lambda m: f'max: {random.randint(8, 15)}', content)
    return content

@modifier_for('example_176.js')
def modify_176(content):
    def repl_force(m):
        new_repulsion = random.randint(40, 80)
        new_edgeLength = random.randint(1, 5)
        return f'repulsion: {new_repulsion},\n        edgeLength: {new_edgeLength}'
    content = re.sub(r'repulsion:\s*\d+,\s*edgeLength:\s*\d+', repl_force, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_180.js')
def modify_180(content):
    def repl_scale(m):
        min_val = random.uniform(0.2, 0.6)
        max_val = random.uniform(1.5, 3.0)
        return f'scaleLimit: {{\n          min: {min_val:.2f},\n          max: {max_val:.2f}\n        }}'
    content = re.sub(r'scaleLimit:\s*\{\s*min:\s*[\d.]+\s*,\s*max:\s*[\d.]+\s*\}', repl_scale, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_181.js')
def modify_181(content):
    return modify_180(content)

@modifier_for('example_182.js')
def modify_182(content):
    return modify_180(content)

@modifier_for('example_186.js')
def modify_186(content):
    def repl_linestyle(m):
        width_str = m.group(1)
        opacity_str = m.group(2)
        new_width = perturb_number(float(width_str), 0.3, 0.2)
        new_opacity = min(1, max(0, float(opacity_str) + random.uniform(-0.2, 0.2)))
        return f'width: {new_width:.2f},\n              curveness: 0.3,\n              opacity: {new_opacity:.2f}'
    content = re.sub(r'width:\s*([\d.]+),\s*curveness:\s*[\d.]+\s*,\s*opacity:\s*([\d.]+)', repl_linestyle, content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_191.js')
def modify_191(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_percent(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.2, 5)
        return f'{new_num:.1f}%'
    content = re.sub(r'(\d+(?:\.\d+)?)%', repl_percent, content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_194.js')
def modify_194(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_fork(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.2, 10)
        return f'edgeForkPosition: "{new_num:.1f}%"'
    content = re.sub(r'edgeForkPosition:\s*"(\d+(?:\.\d+)?)%"', repl_fork, content)
    def repl_width(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 1)
        return f'width: {new_num:.1f}'
    content = re.sub(r'width:\s*(\d+)', repl_width, content)
    return content

@modifier_for('example_199.js')
def modify_199(content):
    def repl_visible(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'visibleMin: {int(new_num)}'
    content = re.sub(r'visibleMin:\s*(\d+)', repl_visible, content)
    def repl_sat(m):
        low = m.group(1)
        high = m.group(2)
        new_low = min(1, max(0, float(low) + random.uniform(-0.1, 0.1)))
        new_high = min(1, max(0, float(high) + random.uniform(-0.1, 0.1)))
        return f'colorSaturation: [{new_low:.2f}, {new_high:.2f}]'
    content = re.sub(r'colorSaturation:\s*\[([\d.]+),\s*([\d.]+)\]', repl_sat, content)
    def repl_border_sat(m):
        num_str = m.group(1)
        new_num = min(1, max(0, float(num_str) + random.uniform(-0.1, 0.1)))
        return f'borderColorSaturation: {new_num:.2f}'
    content = re.sub(r'borderColorSaturation:\s*([\d.]+)', repl_border_sat, content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_202.js')
def modify_202(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 5)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    return content

@modifier_for('example_204.js')
def modify_204(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 3)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_radius(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.2, 10)
        return f'radius: [0, "{new_num:.0f}%"]'
    content = re.sub(r'radius:\s*\[\s*0\s*,\s*\'(\d+)%\'\s*\]', repl_radius, content)
    return content

@modifier_for('example_205.js')
def modify_205(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 3)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_radius(m):
        num1 = m.group(1)
        num2 = m.group(2)
        new_num1 = perturb_number(float(num1), 0.2, 10)
        new_num2 = perturb_number(float(num2), 0.2, 10)
        return f'radius: [{new_num1:.0f}, "{new_num2:.0f}%"]'
    content = re.sub(r'radius:\s*\[\s*(\d+)\s*,\s*\'(\d+)%\'\s*\]', repl_radius, content)
    def repl_border(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 3)
        return f'borderRadius: {int(new_num)}'
    content = re.sub(r'borderRadius:\s*(\d+)', repl_border, content)
    return content

@modifier_for('example_214.js')
def modify_214(content):
    def repl_rawdata(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return 'const rawData = ' + new_data
    content = re.sub(r'const\s+rawData\s*=\s*(\[.*?\]);', repl_rawdata, content, flags=re.DOTALL)
    def repl_grid(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.2, 2)
        return f'const GRID_WIDTH = {new_expr};'
    content = re.sub(r'const\s+GRID_WIDTH\s*=\s*([^;]+);', repl_grid, content)
    def repl_grid2(m):
        expr = m.group(1)
        new_expr = _replace_numbers_in_str(expr, 0.2, 2)
        return f'const GRID_HEIGHT = {new_expr};'
    content = re.sub(r'const\s+GRID_HEIGHT\s*=\s*([^;]+);', repl_grid2, content)
    def repl_colors(m):
        color_str = m.group(1)
        new_color_str = _replace_colors_in_str(color_str)
        return 'color: ' + new_color_str
    content = re.sub(r'color:\s*(\[.*?\])', repl_colors, content, flags=re.DOTALL)
    return content

@modifier_for('example_215.js')
def modify_215(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 2)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    return content

@modifier_for('example_216.js')
def modify_216(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 2)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_curveness(m):
        num_str = m.group(1)
        new_num = min(1, max(0, float(num_str) + random.uniform(-0.2, 0.2)))
        return f'curveness: {new_num:.2f}'
    content = re.sub(r'curveness:\s*([\d.]+)', repl_curveness, content)
    return content

@modifier_for('example_237.js')
def modify_237(content):
    content = _replace_numbers_in_str(content, 0.2, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_247.js')
def modify_247(content):
    def repl_rawdata(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return 'let rawData = ' + new_data
    content = re.sub(r'let\s+rawData\s*=\s*(\[.*?\]);', repl_rawdata, content, flags=re.DOTALL)
    return content

@modifier_for('example_257.js')
def modify_257(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_258.js')
def modify_258(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_259.js')
def modify_259(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_260.js')
def modify_260(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_261.js')
def modify_261(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_262.js')
def modify_262(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_263.js')
def modify_263(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_269.js')
def modify_269(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_272.js')
def modify_272(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 5)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_275.js')
def modify_275(content):
    def repl_colors(m):
        color_str = m.group(1)
        new_color_str = _replace_colors_in_str(color_str)
        return 'const colorList = ' + new_color_str
    content = re.sub(r'const\s+colorList\s*=\s*(\[.*?\]);', repl_colors, content, flags=re.DOTALL)
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]).map', repl_data, content, flags=re.DOTALL)
    return content

@modifier_for('example_276.js')
def modify_276(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data, content, flags=re.DOTALL)
    return content

@modifier_for('example_277.js')
def modify_277(content):
    def repl_random(m):
        return 'Math.random() * ' + str(random.randint(800, 1200))
    content = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', repl_random, content)
    def repl_start(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.2, 10)
        return f'start: {int(new_num)}'
    content = re.sub(r'start:\s*(\d+)', repl_start, content)
    def repl_end(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.2, 10)
        return f'end: {int(new_num)}'
    content = re.sub(r'end:\s*(\d+)', repl_end, content)
    return content

@modifier_for('example_279.js')
def modify_279(content):
    def repl_val(m):
        return 'Math.random() * ' + str(random.randint(800, 1200))
    content = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', repl_val, content)
    def repl_err(m):
        err_str = m.group(1)
        new_err = _replace_numbers_in_str(err_str, 0.3, 20)
        return 'errorData.push(' + new_err + ')'
    content = re.sub(r'errorData\.push\((\[.*?\])\)', repl_err, content, flags=re.DOTALL)
    return content

@modifier_for('example_282.js')
def modify_282(content):
    def repl_const(m):
        const_name = m.group(1)
        num_str = m.group(2)
        new_num = perturb_number(float(num_str), 0.2, 5)
        return f'var {const_name} = {new_num};'
    content = re.sub(r'var\s+(HEIGHT_RATIO|DATA_ZOOM_AUTO_MOVE_THROTTLE|DATA_ZOOM_AUTO_MOVE_SPEED|DATA_ZOOM_AUTO_MOVE_DETECT_AREA_WIDTH)\s*=\s*([\d.]+);', repl_const, content)
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_291.js')
def modify_291(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    def repl_palette(m):
        color_str = m.group(1)
        new_color_str = _replace_colors_in_str(color_str)
        return 'const defaultPalette = ' + new_color_str
    content = re.sub(r'const\s+defaultPalette\s*=\s*(\[.*?\]);', repl_palette, content, flags=re.DOTALL)
    return content

@modifier_for('example_292.js')
def modify_292(content):
    def repl_val(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 5)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_val, content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_294.js')
def modify_294(content):
    return modify_81(content)

@modifier_for('example_295.js')
def modify_295(content):
    def repl_source(m):
        source_str = m.group(1)
        new_source = _replace_numbers_in_str(source_str, 0.3, 5)
        return 'source: ' + new_source
    content = re.sub(r'source:\s*(\[.*?\])', repl_source, content, flags=re.DOTALL)
    content = re.sub(r'min:\s*(\d+)', lambda m: f'min: {random.randint(5, 15)}', content)
    content = re.sub(r'max:\s*(\d+)', lambda m: f'max: {random.randint(90, 110)}', content)
    def repl_colors(m):
        color_str = m.group(1)
        new_color_str = _replace_colors_in_str(color_str)
        return 'color: ' + new_color_str
    content = re.sub(r'color:\s*(\[.*?\])', repl_colors, content, flags=re.DOTALL)
    return content

@modifier_for('example_296.js')
def modify_296(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_297.js')
def modify_297(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_299.js')
def modify_299(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_301.js')
def modify_301(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_302.js')
def modify_302(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_303.js')
def modify_303(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 10)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_307.js')
def modify_307(content):
    def repl_random(m):
        return 'Math.random() * ' + str(random.randint(200, 400))
    content = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', repl_random, content)
    content = re.sub(r'let\s+oneDay\s*=\s*(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+);',
                     lambda m: f'let oneDay = {random.randint(20,28)} * 3600 * 1000;', content)
    content = re.sub(r'let\s+valueBase\s*=\s*Math\.random\(\)\s*\*\s*(\d+);',
                     lambda m: f'let valueBase = Math.random() * {random.randint(200,400)};', content)
    content = re.sub(r'let\s+valueBase2\s*=\s*Math\.random\(\)\s*\*\s*(\d+);',
                     lambda m: f'let valueBase2 = Math.random() * {random.randint(30,70)};', content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_308.js')
def modify_308(content):
    def repl_duration(m):
        return f'duration: {random.randint(2000, 5000)}'
    content = re.sub(r'duration:\s*(\d+)', repl_duration, content)
    def repl_dash(m):
        return f'lineDash: [{random.randint(0,100)}, {random.randint(100,300)}]'
    content = re.sub(r'lineDash:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', repl_dash, content)
    content = re.sub(r'lineWidth:\s*(\d+)', lambda m: f'lineWidth: {random.randint(1,3)}', content)
    return content

@modifier_for('example_309.js')
def modify_309(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_311.js')
def modify_311(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'data: ' + new_data
    content = re.sub(r'data:\s*(\[.*?\])', repl_data, content, flags=re.DOTALL)
    content = _replace_colors_in_str(content)
    content = re.sub(r'rotation:\s*Math\.PI\s*/\s*(\d+)',
                     lambda m: f'rotation: Math.PI / {random.randint(3,6)}', content)
    def repl_shape(m):
        w = m.group(1)
        h = m.group(2)
        new_w = perturb_number(float(w), 0.2, 20)
        new_h = perturb_number(float(h), 0.2, 20)
        return f'shape: {{ width: {int(new_w)}, height: {int(new_h)} }}'
    content = re.sub(r'shape:\s*\{\s*width:\s*(\d+),\s*height:\s*(\d+)\s*\}', repl_shape, content)
    return content

@modifier_for('example_312.js')
def modify_312(content):
    def repl_data(m):
        data_str = m.group(1)
        new_data = _replace_numbers_in_str(data_str, 0.3, 5)
        return 'const data = ' + new_data
    content = re.sub(r'const\s+data\s*=\s*(\[.*?\]);', repl_data, content, flags=re.DOTALL)
    content = re.sub(r'min:\s*(-?\d+)', lambda m: f'min: {random.randint(-120, -80)}', content)
    content = re.sub(r'max:\s*(\d+)', lambda m: f'max: {random.randint(60, 90)}', content)
    content = re.sub(r'symbolSize:\s*(\d+)', lambda m: f'symbolSize: {random.randint(15,25)}', content)
    return content

@modifier_for('example_314.js')
def modify_314(content):
    def repl_value(m):
        num_str = m.group(1)
        new_num = perturb_number(float(num_str), 0.3, 50)
        return f'value: {int(new_num)}'
    content = re.sub(r'value:\s*(\d+)', repl_value, content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_328.js')
def modify_328(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_331.js')
def modify_331(content):
    def repl_visual(m):
        max_val = m.group(1)
        new_max = perturb_number(float(max_val), 0.3, 10)
        return f'max: {int(new_max)}'
    content = re.sub(r'max:\s*(\d+)', repl_visual, content)
    content = re.sub(r'barSize:\s*([\d.]+)', lambda m: f'barSize: {random.uniform(0.4, 0.8):.2f}', content)
    content = re.sub(r'minHeight:\s*([\d.]+)', lambda m: f'minHeight: {random.uniform(0.1, 0.5):.2f}', content)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_347.js')
def modify_347(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_108.js')
def modify_108(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_109.js')
def modify_109(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_110.js')
def modify_110(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_112.js')
def modify_112(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_113.js')
def modify_113(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_114.js')
def modify_114(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_116.js')
def modify_116(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_117.js')
def modify_117(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_123.js')
def modify_123(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_124.js')
def modify_124(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_127.js')
def modify_127(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

@modifier_for('example_128.js')
def modify_128(content):
    content = _replace_numbers_in_str(content, 0.3, 10)
    content = _replace_colors_in_str(content)
    return content

# ==================== 主处理函数 ====================

def process_one_file(args):
    input_path, output_dir, num_variations, seed = args
    random.seed(seed)
    base_name = Path(input_path).stem
    filename = Path(input_path).name
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件失败 {input_path}: {e}")
        return f"失败 {base_name}"

    # 判断是否有定制修改器
    if filename in MODIFIERS:
        modifier = MODIFIERS[filename]
        for i in range(num_variations):
            random.seed(seed + i)
            new_content = modifier(content)
            out_path = os.path.join(output_dir, f"{base_name}_aug_{i:03d}.js")
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"写入文件失败 {out_path}: {e}")
    else:
        # 使用通用增强
        new_contents = general_augment(content, num_variations, seed)
        for i, new_content in enumerate(new_contents):
            out_path = os.path.join(output_dir, f"{base_name}_aug_{i:03d}.js")
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"写入文件失败 {out_path}: {e}")

    return f"完成 {base_name} 生成 {num_variations} 个变体"

def main():
    parser = argparse.ArgumentParser(description="统一 ECharts JS 增强脚本")
    parser.add_argument('--input_dir', type=str, required=True, help='原始ECharts JS文件所在目录')
    parser.add_argument('--output_dir', type=str, default='./output/echarts_augmented', help='增强后JS输出目录')
    parser.add_argument('--num', type=int, default=5, help='每个文件生成多少个变体')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    args = parser.parse_args()

    set_random_seed(args.seed)
    input_dir = Path(args.input_dir)
    js_files = list(input_dir.glob('*.js'))
    print(f"找到 {len(js_files)} 个JS文件")

    task_args = [(str(f), args.output_dir, args.num, args.seed + i) for i, f in enumerate(js_files)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one_file, arg) for arg in task_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="增强进度"):
            try:
                result = future.result()
                # 可选打印结果
                # print(result)
            except Exception as e:
                print(f"处理失败: {e}")

if __name__ == '__main__':
    main()