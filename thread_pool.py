"""
多线程处理模块
使用线程池并行处理图片检测任务
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Dict
from tqdm import tqdm
import logging

from config import DEFAULT_THREAD_COUNT, MAX_THREAD_COUNT

logger = logging.getLogger(__name__)


class ThreadPoolManager:
    """线程池管理器类"""

    def __init__(self, num_threads: int = None):
        """
        初始化线程池管理器

        Args:
            num_threads: 线程数，如果为None则使用默认值
        """
        if num_threads is None:
            self.num_threads = DEFAULT_THREAD_COUNT
        else:
            self.num_threads = min(num_threads, MAX_THREAD_COUNT)

        logger.info(f"Initialized ThreadPoolManager with {self.num_threads} threads")

    def process_images(
        self,
        image_list: List[str],
        detector,
        show_progress: bool = True,
        on_result_callback: Callable[[Dict], None] = None,
    ) -> List[Dict]:
        """
        并行处理图片列表

        Args:
            image_list: 图片文件路径列表
            detector: 折角检测器实例
            show_progress: 是否显示进度条
            on_result_callback: 结果回调函数，在每个图片处理完成后调用

        Returns:
            检测结果列表
        """
        results = []

        if not image_list:
            logger.warning("No images to process")
            return results

        logger.info(
            f"Processing {len(image_list)} images with {self.num_threads} threads"
        )

        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有任务
            future_to_image = {
                executor.submit(detector.detect_fold, image_path): image_path
                for image_path in image_list
            }

            # 使用进度条显示处理进度
            if show_progress:
                progress_bar = tqdm(
                    total=len(image_list), desc="Detecting folds", unit="image"
                )

            # 收集结果
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)

                    # 调用回调函数（如果提供了）
                    if on_result_callback:
                        try:
                            on_result_callback(result)
                        except Exception as e:
                            logger.error(f"Callback error for {image_path}: {str(e)}")

                    if show_progress:
                        progress_bar.update(1)

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    error_result = {
                        "image_path": image_path,
                        "has_fold": False,
                        "confidence": 0.0,
                        "folded_corners": [],
                        "error": str(e),
                    }
                    results.append(error_result)

                    if show_progress:
                        progress_bar.update(1)

            if show_progress:
                progress_bar.close()

        logger.info(f"Completed processing {len(results)} images")
        return results

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        计算检测结果统计信息

        Args:
            results: 检测结果列表

        Returns:
            统计信息字典
        """
        total = len(results)
        folded = sum(1 for r in results if r["has_fold"])
        errors = sum(1 for r in results if r.get("error"))

        stats = {
            "total_images": total,
            "folded_images": folded,
            "normal_images": total - folded - errors,
            "error_images": errors,
            "fold_rate": folded / total if total > 0 else 0.0,
        }

        return stats
