"""
图片折角检测核心模块
使用OpenCV进行图像处理和折角检测
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

from config import DETECTION_PARAMS

logger = logging.getLogger(__name__)


class FoldDetector:
    """折角检测器类"""

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化折角检测器

        Args:
            params: 检测参数字典，如果为None则使用默认配置
        """
        self.params = params or DETECTION_PARAMS

    def detect_fold(self, image_path: str) -> Dict:
        """
        检测单张图片是否有折角

        Args:
            image_path: 图片文件路径

        Returns:
            检测结果字典，包含：
            - has_fold: bool，是否检测到折角
            - confidence: float，置信度（0-1）
            - folded_corners: list，检测到折角的位置列表
            - error: str，错误信息（如有）
        """
        result = {
            "image_path": image_path,
            "has_fold": False,
            "confidence": 0.0,
            "folded_corners": [],
            "error": None,
        }

        try:
            # 使用numpy读取文件以支持中文路径
            # cv2.imread()不支持中文路径，所以使用这种方法
            image_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                result["error"] = "Failed to load image"
                return result

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测每个角落
            corners = self._check_all_corners(gray, image.shape)
            result["folded_corners"] = corners

            # 判断是否有折角
            if len(corners) > 0:
                result["has_fold"] = True
                result["confidence"] = min(len(corners) * 0.3, 1.0)

        except Exception as e:
            logger.error(f"Error detecting fold in {image_path}: {str(e)}")
            result["error"] = str(e)

        return result

    def _check_all_corners(self, gray_image: np.ndarray, shape: Tuple) -> list:
        """
        检查图片的所有四个角落

        Args:
            gray_image: 灰度图像
            shape: 图像形状

        Returns:
            检测到折角的位置列表 ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        """
        height, width = gray_image.shape[:2]
        corner_size = int(min(height, width) * self.params["corner_region_ratio"])

        folded_corners = []

        # 定义四个角落的位置
        corners = {
            "top_left": (0, 0, corner_size, corner_size),
            "top_right": (width - corner_size, 0, width, corner_size),
            "bottom_left": (0, height - corner_size, corner_size, height),
            "bottom_right": (width - corner_size, height - corner_size, width, height),
        }

        # 检查每个角落
        for corner_name, (x1, y1, x2, y2) in corners.items():
            corner_region = gray_image[y1:y2, x1:x2]
            if self._detect_fold_in_region(corner_region):
                folded_corners.append(corner_name)

        return folded_corners

    def _detect_fold_in_region(self, region: np.ndarray) -> bool:
        """
        在指定区域检测折角

        Args:
            region: 图像区域（灰度）

        Returns:
            是否检测到折角
        """
        try:
            # 边缘检测
            edges = cv2.Canny(
                region, self.params["canny_threshold1"], self.params["canny_threshold2"]
            )

            # 查找轮廓
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 计算最小轮廓面积阈值
            min_area = (
                region.shape[0]
                * region.shape[1]
                * self.params["min_contour_area_ratio"]
            )

            # 检查是否有显著的折角轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # 使用多边形拟合
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # 如果近似为三角形，可能是折角
                    if len(approx) == 3:
                        return True

            # 角点检测
            corners = cv2.cornerHarris(
                region.astype(np.float32),
                self.params["harris_block_size"],
                self.params["harris_ksize"],
                self.params["harris_k"],
            )

            # 归一化角点响应
            corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)

            # 如果角点过多，可能存在异常
            corner_threshold = 200
            strong_corners = np.sum(corners > corner_threshold)

            if strong_corners > 5:
                return True

            return False

        except Exception as e:
            logger.warning(f"Error in region fold detection: {str(e)}")
            return False
