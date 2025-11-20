"""
配置文件
包含图片折角检测的默认配置参数
"""

import os
from multiprocessing import cpu_count

# 多线程配置
DEFAULT_THREAD_COUNT = cpu_count()  # 默认使用CPU核心数
MAX_THREAD_COUNT = 16  # 最大线程数限制

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".jfif",
}

# 折角检测参数
DETECTION_CONFIG = {
    # 角点检测参数
    "harris_block_size": 2,
    "harris_ksize": 3,
    "harris_k": 0.04,
    "harris_threshold": 0.01,
    # 边缘检测参数（Canny）
    "canny_threshold1": 50,
    "canny_threshold2": 150,
    "canny_aperture": 3,
    # 折角判断阈值
    "corner_region_ratio": 0.15,  # 角落区域大小（相对于图片尺寸的比例）
    "fold_angle_threshold": 20,  # 折角角度阈值（度）
    "min_contour_area_ratio": 0.01,  # 最小轮廓面积比例
    "fold_confidence_threshold": 0.4,  # 折角检测置信度阈值（0-1），越高越严格
    # 图像预处理
    "blur_kernel_size": (5, 5),
    "resize_max_dimension": 1024,  # 为提高处理速度，限制最大尺寸
}

# 输出配置
OUTPUT_CONFIG = {
    "show_progress": True,
    "verbose": False,
    "save_detection_images": False,  # 是否保存检测可视化图片
    "output_dir": "detection_results",
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # None表示不保存日志文件，可设置为文件路径
}

# 为了兼容性，添加直接变量
LOG_LEVEL = LOGGING_CONFIG["level"]
LOG_FORMAT = LOGGING_CONFIG["format"]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SHOW_PROGRESS_BAR = OUTPUT_CONFIG["show_progress"]
VERBOSE_OUTPUT = OUTPUT_CONFIG["verbose"]

# 检测参数（用于fold_detector.py）
DETECTION_PARAMS = DETECTION_CONFIG
