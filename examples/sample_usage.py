"""
示例使用脚本
演示如何使用FoldDetect进行图片折角检测
"""

from fold_detector import FoldDetector
from image_scanner import ImageScanner
from thread_pool import ThreadPoolManager


def example_basic_detection():
    """基本检测示例"""
    print("=" * 60)
    print("示例1: 基本折角检测")
    print("=" * 60)

    # 创建检测器
    detector = FoldDetector()

    # 检测单张图片
    result = detector.detect_fold("test_images/sample.jpg")

    print(f"图片路径: {result['image_path']}")
    print(f"是否有折角: {result['has_fold']}")
    print(f"置信度: {result['confidence']:.2f}")
    if result["folded_corners"]:
        print(f"折角位置: {', '.join(result['folded_corners'])}")
    if result.get("error"):
        print(f"错误: {result['error']}")
    print()


def example_batch_detection():
    """批量检测示例"""
    print("=" * 60)
    print("示例2: 批量检测（多线程）")
    print("=" * 60)

    # 扫描目录
    scanner = ImageScanner()
    image_files = scanner.scan_directory("test_images", recursive=True)

    print(f"找到 {len(image_files)} 张图片")

    # 创建检测器和线程池管理器
    detector = FoldDetector()
    thread_manager = ThreadPoolManager(num_threads=4)

    # 批量检测
    results = thread_manager.process_images(image_files, detector, show_progress=True)

    # 显示统计
    stats = thread_manager.get_statistics(results)
    print(f"\n统计结果:")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  检测到折角: {stats['folded_images']}")
    print(f"  正常图片: {stats['normal_images']}")
    print(f"  折角比例: {stats['fold_rate']:.2%}")
    print()


def example_custom_parameters():
    """自定义参数示例"""
    print("=" * 60)
    print("示例3: 自定义检测参数")
    print("=" * 60)

    # 自定义检测参数
    custom_params = {
        "canny_threshold1": 30,
        "canny_threshold2": 100,
        "harris_block_size": 2,
        "harris_ksize": 3,
        "harris_k": 0.04,
        "corner_region_ratio": 0.2,  # 增大角落检测区域
        "fold_threshold": 0.3,
        "min_contour_area_ratio": 0.005,
    }

    # 使用自定义参数创建检测器
    detector = FoldDetector(params=custom_params)

    # 检测图片
    result = detector.detect_fold("test_images/sample.jpg")

    print(f"使用自定义参数检测结果:")
    print(f"  有折角: {result['has_fold']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print()


if __name__ == "__main__":
    print("\nFoldDetect 使用示例\n")

    # 运行示例
    try:
        example_basic_detection()
    except Exception as e:
        print(f"示例1执行出错: {e}\n")

    try:
        example_batch_detection()
    except Exception as e:
        print(f"示例2执行出错: {e}\n")

    try:
        example_custom_parameters()
    except Exception as e:
        print(f"示例3执行出错: {e}\n")

    print("示例演示完成！")
