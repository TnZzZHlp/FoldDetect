"""
图片扫描模块
递归扫描目录并过滤图片文件
"""

import os
from pathlib import Path
from typing import List
import logging

from config import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


class ImageScanner:
    """图片扫描器类"""

    def __init__(self, supported_formats: List[str] = None):
        """
        初始化图片扫描器

        Args:
            supported_formats: 支持的图片格式列表，如果为None则使用默认配置
        """
        self.supported_formats = supported_formats or SUPPORTED_IMAGE_FORMATS

    def scan_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """
        扫描指定目录，返回所有支持的图片文件路径

        Args:
            directory_path: 目录路径
            recursive: 是否递归扫描子目录

        Returns:
            图片文件路径列表
        """
        image_files = []

        try:
            directory = Path(directory_path)

            if not directory.exists():
                logger.error(f"Directory does not exist: {directory_path}")
                return image_files

            if not directory.is_dir():
                logger.error(f"Path is not a directory: {directory_path}")
                return image_files

            # 扫描文件
            if recursive:
                # 递归扫描所有子目录
                for file_path in directory.rglob("*"):
                    if self._is_supported_image(file_path):
                        image_files.append(str(file_path.absolute()))
            else:
                # 只扫描当前目录
                for file_path in directory.glob("*"):
                    if self._is_supported_image(file_path):
                        image_files.append(str(file_path.absolute()))

            logger.info(f"Found {len(image_files)} image(s) in {directory_path}")

        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {str(e)}")

        return image_files

    def _is_supported_image(self, file_path: Path) -> bool:
        """
        检查文件是否为支持的图片格式

        Args:
            file_path: 文件路径

        Returns:
            是否为支持的图片格式
        """
        if not file_path.is_file():
            return False

        extension = file_path.suffix.lower()
        return extension in self.supported_formats

    def filter_images(self, file_list: List[str]) -> List[str]:
        """
        从文件列表中过滤出支持的图片文件

        Args:
            file_list: 文件路径列表

        Returns:
            图片文件路径列表
        """
        return [
            file_path
            for file_path in file_list
            if self._is_supported_image(Path(file_path))
        ]
