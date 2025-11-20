"""
测试折角检测功能
用于验证优化后的折角检测算法
"""

import cv2
import numpy as np
from pathlib import Path
import logging

from fold_detector import FoldDetector
from config import DETECTION_PARAMS

# 配置日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_single_image(image_path: str):
    """
    测试单张图片的折角检测

    Args:
        image_path: 图片路径
    """
    print(f"\n{'='*60}")
    print(f"正在测试图片: {image_path}")
    print(f"{'='*60}\n")

    # 创建检测器
    detector = FoldDetector()

    # 检测折角
    result = detector.detect_fold(image_path)

    # 打印结果
    print("检测结果:")
    print(f"  - 图片路径: {result['image_path']}")
    print(f"  - 是否有折角: {result['has_fold']}")
    print(f"  - 置信度: {result['confidence']:.2f}")
    print(
        f"  - 折角位置: {', '.join(result['folded_corners']) if result['folded_corners'] else '无'}"
    )

    if result["error"]:
        print(f"  - 错误: {result['error']}")

    # 可视化检测结果（可选）
    try:
        # 读取图片
        image_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is not None:
            # 在图片上标记检测到的折角
            height, width = image.shape[:2]
            corner_size = int(
                min(height, width) * DETECTION_PARAMS["corner_region_ratio"]
            )

            # 定义四个角落的位置
            corners_coords = {
                "top_left": (0, 0, corner_size, corner_size),
                "top_right": (width - corner_size, 0, width, corner_size),
                "bottom_left": (0, height - corner_size, corner_size, height),
                "bottom_right": (
                    width - corner_size,
                    height - corner_size,
                    width,
                    height,
                ),
            }

            # 标记检测到折角的区域
            for corner_name in result["folded_corners"]:
                if corner_name in corners_coords:
                    x1, y1, x2, y2 = corners_coords[corner_name]
                    # 画矩形框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # 添加文字标签
                    label = corner_name.replace("_", " ").title()
                    cv2.putText(
                        image,
                        label,
                        (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # 在图片上显示总体结果
            status_text = f"Fold Detected: {result['has_fold']} (Conf: {result['confidence']:.2f})"
            cv2.putText(
                image,
                status_text,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if result["has_fold"] else (255, 0, 0),
                2,
            )

            # 保存结果图片
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)

            input_filename = Path(image_path).stem
            output_path = output_dir / f"{input_filename}_result.jpg"

            # 使用imencode保存以支持中文路径
            _, encoded_image = cv2.imencode(".jpg", image)
            encoded_image.tofile(str(output_path))

            print(f"\n可视化结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n可视化过程出错: {str(e)}")

    print(f"\n{'='*60}\n")
    return result


def test_multiple_images(image_dir: str):
    """
    测试目录中的所有图片

    Args:
        image_dir: 图片目录
    """
    from image_scanner import ImageScanner

    print(f"\n扫描目录: {image_dir}")
    scanner = ImageScanner()
    image_files = scanner.scan_directory(image_dir)

    print(f"找到 {len(image_files)} 张图片\n")

    results = []
    for image_path in image_files:
        result = test_single_image(str(image_path))
        results.append(result)

    # 统计结果
    total = len(results)
    with_fold = sum(1 for r in results if r["has_fold"])
    without_fold = total - with_fold

    print("\n" + "=" * 60)
    print("总体统计:")
    print(f"  - 总图片数: {total}")
    print(f"  - 检测到折角: {with_fold}")
    print(f"  - 未检测到折角: {without_fold}")
    print(f"  - 折角比例: {with_fold/total*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  测试单张图片: python test_fold_detection.py <图片路径>")
        print("  测试目录: python test_fold_detection.py <目录路径>")
        sys.exit(1)

    input_path = sys.argv[1]
    path = Path(input_path)

    if path.is_file():
        test_single_image(str(path))
    elif path.is_dir():
        test_multiple_images(str(path))
    else:
        print(f"错误: 路径不存在或无效: {input_path}")
        sys.exit(1)
