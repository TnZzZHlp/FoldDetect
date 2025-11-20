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
        在指定区域检测折角（严格模式 - 降低误报率）

        Args:
            region: 图像区域（灰度）

        Returns:
            是否检测到折角
        """
        try:
            if region.size == 0:
                return False

            # 预处理：降噪
            denoised = cv2.GaussianBlur(region, (5, 5), 0)

            # 核心检测：三角形区域检测（折角最可靠的特征）
            triangle_score = self._detect_triangle_fold(denoised)

            # 如果没有检测到三角形，直接返回False
            # 真正的折角必然会有三角形特征
            if triangle_score < 0.5:
                logger.debug(
                    f"No triangle detected (score: {triangle_score:.2f}), not a fold"
                )
                return False

            # 辅助验证：边缘检测
            edge_score = self._detect_fold_edges(denoised)

            # 禁用容易误报的特征
            # shadow_score角点检测会被定位标记点误触发
            shadow_score = 0  # 禁用阴影检测
            corner_score = 0  # 禁用角点检测
            texture_score = 0  # 禁用纹理检测（也容易误报）

            # 简化的评分系统 - 只使用三角形和边缘
            fold_score = triangle_score * 3.0 + edge_score * 1.0  # 三角形权重更高
            max_score = 4.0
            confidence = fold_score / max_score

            # 更严格的阈值
            threshold = self.params.get("fold_confidence_threshold", 0.75)

            logger.debug(
                f"Fold detection scores - Triangle: {triangle_score:.2f}, Edge: {edge_score:.2f}, "
                f"Confidence: {confidence:.2f}, Threshold: {threshold:.2f}"
            )

            return confidence > threshold

        except Exception as e:
            logger.warning(f"Error in region fold detection: {str(e)}")
            return False

    def _detect_shadow(self, region: np.ndarray) -> float:
        """
        检测阴影区域（折角会产生阴影）

        Returns:
            0-1的得分，1表示检测到明显阴影
        """
        try:
            # 计算图像的暗区比例
            mean_val = np.mean(region)
            std_val = np.std(region)

            # 找到比平均值暗很多的区域
            dark_threshold = mean_val - std_val * 0.5
            dark_pixels = np.sum(region < dark_threshold)
            dark_ratio = dark_pixels / region.size

            # 检测三角形暗区（折角特征）
            # 使用阈值分割
            _, binary = cv2.threshold(
                region, int(dark_threshold), 255, cv2.THRESH_BINARY_INV
            )

            # 查找暗区轮廓
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                # 找最大的暗区
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # 如果暗区面积足够大且形状接近三角形
                if area > region.size * 0.05:  # 至少5%的区域
                    # 检查是否为三角形
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        # 三角形的实心度通常较高
                        if solidity > 0.7:
                            return min(dark_ratio * 3, 1.0)

            return min(dark_ratio * 2, 1.0)

        except Exception as e:
            logger.debug(f"Shadow detection error: {str(e)}")
            return 0.0

    def _detect_fold_edges(self, region: np.ndarray) -> float:
        """
        多尺度边缘检测

        Returns:
            0-1的得分
        """
        try:
            # 使用自适应Canny阈值
            median_val = np.median(region)
            lower = int(max(0, 0.7 * median_val))
            upper = int(min(255, 1.3 * median_val))

            edges = cv2.Canny(region, lower, upper)

            # 查找轮廓
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                return 0.0

            # 计算最小轮廓面积阈值
            min_area = (
                region.shape[0]
                * region.shape[1]
                * self.params["min_contour_area_ratio"]
            )

            # 检查显著轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # 使用多边形拟合
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # 三角形或四边形可能是折角
                    if 3 <= len(approx) <= 5:
                        return 1.0

            # 边缘密度
            edge_density = np.sum(edges > 0) / edges.size
            return min(edge_density * 10, 1.0)

        except Exception as e:
            logger.debug(f"Edge detection error: {str(e)}")
            return 0.0

    def _detect_triangle_fold(self, region: np.ndarray) -> float:
        """
        检测三角形折角（折角最典型的特征）

        Returns:
            0-1的得分
        """
        try:
            # 自适应阈值
            binary = cv2.adaptiveThreshold(
                region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            # 形态学操作，连接断裂的边缘
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            h, w = region.shape
            corner_area = h * w

            for contour in contours:
                area = cv2.contourArea(contour)

                # 面积在合理范围内
                if area < corner_area * 0.03 or area > corner_area * 0.5:
                    continue

                # 多边形拟合
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 检查是否为三角形
                if len(approx) == 3:
                    # 验证三角形是否在角落位置
                    # 计算质心
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 检查顶点是否在角落
                        vertices = approx.reshape(-1, 2)
                        corner_vertices = 0

                        # 统计在边缘的顶点数量
                        margin = min(h, w) * 0.1
                        for vx, vy in vertices:
                            if (vx < margin or vx > w - margin) and (
                                vy < margin or vy > h - margin
                            ):
                                corner_vertices += 1

                        # 如果至少有2个顶点在角落区域
                        if corner_vertices >= 2:
                            return 1.0

                # 也检查近似四边形（有时折角会检测为四边形）
                elif len(approx) == 4:
                    # 计算凸性
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.8:  # 比较实心
                            return 0.7  # 给一个较高但不是最高的分数

            return 0.0

        except Exception as e:
            logger.debug(f"Triangle detection error: {str(e)}")
            return 0.0

    def _analyze_texture_change(self, region: np.ndarray) -> float:
        """
        分析纹理变化（折角区域纹理会发生变化）

        Returns:
            0-1的得分
        """
        try:
            # 使用局部方差分析纹理
            # 计算局部标准差
            kernel_size = max(3, min(region.shape) // 10)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # 使用Sobel算子检测纹理变化
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # 计算梯度的标准差
            gradient_std = np.std(gradient_magnitude)
            gradient_mean = np.mean(gradient_magnitude)

            # 归一化
            if gradient_mean > 0:
                texture_score = min(gradient_std / gradient_mean, 1.0)
            else:
                texture_score = 0.0

            # 检测纹理不连续区域
            # 将区域分成几个子区域，比较方差
            h, w = region.shape
            mid_h, mid_w = h // 2, w // 2

            # 四个子区域
            regions = [
                region[:mid_h, :mid_w],
                region[:mid_h, mid_w:],
                region[mid_h:, :mid_w],
                region[mid_h:, mid_w:],
            ]

            variances = [np.std(r) for r in regions if r.size > 0]

            if len(variances) > 1:
                variance_diff = np.std(variances)
                # 如果不同子区域方差差异大，可能有折角
                discontinuity_score = min(variance_diff / 50, 1.0)
                texture_score = max(texture_score, discontinuity_score)

            return texture_score

        except Exception as e:
            logger.debug(f"Texture analysis error: {str(e)}")
            return 0.0

    def _detect_corners(self, region: np.ndarray) -> float:
        """
        角点检测（原有方法的改进版）

        Returns:
            0-1的得分
        """
        try:
            # Harris角点检测
            corners = cv2.cornerHarris(
                region.astype(np.float32),
                self.params["harris_block_size"],
                self.params["harris_ksize"],
                self.params["harris_k"],
            )

            # 归一化
            corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)

            # 自适应阈值
            threshold = np.mean(corners) + 2 * np.std(corners)
            strong_corners = np.sum(corners > threshold)

            # 归一化得分
            # 正常图像角落应该有1-3个强角点
            # 折角会产生额外的角点
            if strong_corners > 5:
                return min((strong_corners - 3) / 10, 1.0)

            return 0.0

        except Exception as e:
            logger.debug(f"Corner detection error: {str(e)}")
            return 0.0
