"""
定位点检测模块
检测图片四角的定位标记点（黑色方块）
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MarkerDetector:
    """定位点检测器类"""

    def __init__(
        self,
        corner_region_ratio: float = 0.15,
        min_marker_area: int = 100,
        max_marker_area: int = 10000,
        marker_color_threshold: int = 80,
    ):
        """
        初始化定位点检测器

        Args:
            corner_region_ratio: 角落检测区域比例（默认0.15，即15%）
            min_marker_area: 最小定位点面积（像素）
            max_marker_area: 最大定位点面积（像素）
            marker_color_threshold: 定位点颜色阈值（0-255，越小越接近黑色）
        """
        self.corner_region_ratio = corner_region_ratio
        self.min_marker_area = min_marker_area
        self.max_marker_area = max_marker_area
        self.marker_color_threshold = marker_color_threshold

    def detect_markers(self, image_path: str) -> Dict:
        """
        检测图片四角的定位点

        Args:
            image_path: 图片文件路径

        Returns:
            检测结果字典，包含：
            - all_markers_found: bool，是否找到所有4个定位点
            - found_markers: list，找到的定位点位置
            - missing_markers: list，缺失的定位点位置
            - marker_details: dict，每个定位点的详细信息
            - error: str，错误信息（如有）
        """
        result = {
            "image_path": image_path,
            "all_markers_found": False,
            "found_markers": [],
            "missing_markers": [],
            "marker_details": {},
            "error": None,
        }

        try:
            # 使用numpy读取文件以支持中文路径
            image_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                result["error"] = "Failed to load image"
                return result

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测四个角落的定位点
            corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
            for corner in corners:
                marker_info = self._detect_marker_in_corner(gray, corner)

                if marker_info["found"]:
                    result["found_markers"].append(corner)
                    result["marker_details"][corner] = marker_info
                else:
                    result["missing_markers"].append(corner)
                    result["marker_details"][corner] = marker_info

            # 判断是否找到所有定位点
            result["all_markers_found"] = len(result["found_markers"]) == 4

        except Exception as e:
            logger.error(f"Error detecting markers in {image_path}: {str(e)}")
            result["error"] = str(e)

        return result

    def _detect_marker_in_corner(self, gray_image: np.ndarray, corner: str) -> Dict:
        """
        在指定角落检测定位点

        Args:
            gray_image: 灰度图像
            corner: 角落位置 ('top_left', 'top_right', 'bottom_left', 'bottom_right')

        Returns:
            定位点信息字典
        """
        height, width = gray_image.shape
        corner_size = int(min(height, width) * self.corner_region_ratio)

        # 定义角落区域
        if corner == "top_left":
            region = gray_image[0:corner_size, 0:corner_size]
            offset_x, offset_y = 0, 0
        elif corner == "top_right":
            region = gray_image[0:corner_size, width - corner_size : width]
            offset_x, offset_y = width - corner_size, 0
        elif corner == "bottom_left":
            region = gray_image[height - corner_size : height, 0:corner_size]
            offset_x, offset_y = 0, height - corner_size
        else:  # bottom_right
            region = gray_image[
                height - corner_size : height, width - corner_size : width
            ]
            offset_x, offset_y = width - corner_size, height - corner_size

        # 二值化：提取黑色区域
        _, binary = cv2.threshold(
            region, self.marker_color_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 查找符合条件的定位点
        best_marker = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < self.min_marker_area or area > self.max_marker_area:
                continue

            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算方形度（接近1表示是正方形）
            aspect_ratio = float(w) / h if h > 0 else 0
            squareness = min(aspect_ratio, 1 / aspect_ratio) if aspect_ratio > 0 else 0

            # 计算填充度（轮廓面积/边界框面积）
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0

            # 评分：方形度和填充度的综合
            score = squareness * 0.5 + fill_ratio * 0.5

            # 选择得分最高的作为定位点
            if score > best_score and score > 0.6:  # 阈值0.6
                best_score = score
                best_marker = {
                    "found": True,
                    "area": int(area),
                    "position": (int(x + offset_x), int(y + offset_y)),
                    "size": (int(w), int(h)),
                    "squareness": float(squareness),
                    "fill_ratio": float(fill_ratio),
                    "score": float(score),
                }

        if best_marker is None:
            return {
                "found": False,
                "reason": "No suitable marker found in corner region",
            }

        return best_marker

    def visualize_detection(self, image_path: str, output_path: str = None) -> bool:
        """
        可视化定位点检测结果

        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（如果为None则显示而不保存）

        Returns:
            是否成功
        """
        try:
            # 读取图片
            image_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return False

            # 检测定位点
            result = self.detect_markers(image_path)

            # 在图片上标记检测结果
            for corner, info in result["marker_details"].items():
                if info["found"]:
                    x, y = info["position"]
                    w, h = info["size"]

                    # 绘制绿色矩形框（找到的定位点）
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(
                        image,
                        f"{corner}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # 在图片上显示总体结果
            status_text = (
                "All 4 markers found!"
                if result["all_markers_found"]
                else f"Found {len(result['found_markers'])}/4 markers"
            )
            color = (0, 255, 0) if result["all_markers_found"] else (0, 0, 255)
            cv2.putText(
                image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

            # 保存或显示
            if output_path:
                cv2.imencode(".jpg", image)[1].tofile(output_path)
            else:
                cv2.imshow("Marker Detection", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return True

        except Exception as e:
            logger.error(f"Error visualizing detection: {str(e)}")
            return False
