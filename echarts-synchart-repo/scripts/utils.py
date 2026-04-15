# utils.py
import random
import colorsys
import re
from config import COLOR_HUE_VARIATION, COLOR_SATURATION_VARIATION, COLOR_LIGHTNESS_VARIATION

def set_random_seed(seed):
    random.seed(seed)

def random_color_variation(hex_color, hue_var=COLOR_HUE_VARIATION,
                            sat_var=COLOR_SATURATION_VARIATION,
                            light_var=COLOR_LIGHTNESS_VARIATION):
    """
    对给定的十六进制颜色进行随机变化，返回新的十六进制颜色。
    输入格式: '#RRGGBB'
    """
    if not re.match(r'^#[0-9A-Fa-f]{6}$', hex_color):
        return hex_color  # 不是标准颜色，保持不变
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # 随机偏移
    h = (h + random.uniform(-hue_var/360.0, hue_var/360.0)) % 1.0
    s = max(0, min(1, s + random.uniform(-sat_var, sat_var)))
    l = max(0, min(1, l + random.uniform(-light_var, light_var)))

    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    r_new = int(r_new * 255)
    g_new = int(g_new * 255)
    b_new = int(b_new * 255)
    return f'#{r_new:02x}{g_new:02x}{b_new:02x}'

def perturb_number(value, relative_range=0.3, absolute_range=10):
    """
    对数值进行随机扰动。
    如果 value 是整数，返回整数；否则返回浮点数。
    """
    if isinstance(value, int):
        delta = random.randint(-absolute_range, absolute_range)
        return max(0, value + delta)
    else:
        delta = value * random.uniform(-relative_range, relative_range)
        return max(0, value + delta)

def is_solid_color_image(image_path, threshold=0.99):
    """
    判断一张图片是否几乎为纯色（用于过滤空白图）
    需要PIL库，如果不存在则返回False（跳过过滤）
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert('RGB')
        data = np.array(img)
        # 计算唯一颜色数量
        unique_colors = np.unique(data.reshape(-1, 3), axis=0)
        if len(unique_colors) <= 2:  # 背景色可能有两种（抗锯齿）
            # 计算主颜色占比
            counts = np.bincount(data.reshape(-1, 3).dot([1,256,256]))
            main_color_ratio = counts.max() / counts.sum()
            return main_color_ratio > threshold
    except:
        pass
    return False