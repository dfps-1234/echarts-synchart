# config.py
# 增强参数配置

# 颜色修改范围
COLOR_HUE_VARIATION = 30          # 色相偏移量（度）
COLOR_SATURATION_VARIATION = 0.2  # 饱和度随机变化幅度
COLOR_LIGHTNESS_VARIATION = 0.2   # 亮度随机变化幅度

# 数值修改范围
VALUE_RELATIVE_RANGE = 0.1  # 数值相对变化范围（±10.0%）
VALUE_ABSOLUTE_RANGE = 10         # 数值绝对变化范围（整数时使用）

# 文本标签修改
TEXT_LABELS_POOL = [
    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉'],
    # 可继续添加更多文本池
]

# 过滤条件
EMPTY_IMAGE_THRESHOLD = 0.99       # 如果图像中99%以上像素为同一颜色，视为空白图
MIN_OBJECTS = 1                    # 图表中至少包含的图元数量

# 随机种子（可选，保证可复现）
RANDOM_SEED = 42