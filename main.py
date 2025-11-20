"""
图片折角检测主程序
支持多线程处理指定目录下的图片
"""

import argparse
import logging
import sys
import shutil
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from config import (
    DEFAULT_THREAD_COUNT,
    SHOW_PROGRESS_BAR,
    VERBOSE_OUTPUT,
    LOG_LEVEL,
    LOG_FORMAT,
    DATE_FORMAT,
)
from fold_detector import FoldDetector
from image_scanner import ImageScanner
from thread_pool import ThreadPoolManager


def setup_logging(verbose: bool = False):
    """配置日志系统"""
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL)
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="检测指定目录下图片是否有折角（支持多线程处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -d ./images                           # 检测images目录下的所有图片
  python main.py -d ./images -t 8                      # 使用8个线程
  python main.py -d ./images --no-recursive            # 不递归子目录
  python main.py -d ./images -v                        # 详细输出模式
  python main.py -d ./images -o results.txt            # 保存结果到文件
  python main.py -d ./images --target ./folded         # 将折角图片移动到folded目录
  python main.py -d ./images --target ./f --visualize  # 移动并生成可视化图片
  python main.py -d ./images --target ./f --instant-move --visualize  # 即时移动+可视化
        """,
    )

    parser.add_argument(
        "-d", "--directory", type=str, required=True, help="要检测的图片目录路径"
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_THREAD_COUNT,
        help=f"线程数 (默认: {DEFAULT_THREAD_COUNT})",
    )

    parser.add_argument("--no-recursive", action="store_true", help="不递归扫描子目录")

    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出模式")

    parser.add_argument("--no-progress", action="store_true", help="不显示进度条")

    parser.add_argument("-o", "--output", type=str, help="输出结果文件路径（可选）")

    parser.add_argument(
        "--target", type=str, help="将检测到折角的图片移动到指定目录（可选）"
    )

    parser.add_argument(
        "--visualize", action="store_true", help="生成可视化图片（标注折角位置）"
    )

    parser.add_argument(
        "--instant-move",
        action="store_true",
        help="检测到折角后立即移动（而不是等待全部完成）",
    )

    return parser.parse_args()


def visualize_fold_detection(image_path: str, result: dict, output_path: str) -> bool:
    """
    生成折角检测的可视化图片

    Args:
        image_path: 原始图片路径
        result: 检测结果字典
        output_path: 输出图片路径

    Returns:
        是否成功生成可视化图片
    """
    try:
        from config import DETECTION_PARAMS

        # 读取图片
        image_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return False

        height, width = image.shape[:2]
        corner_size = int(min(height, width) * DETECTION_PARAMS["corner_region_ratio"])

        # 定义四个角落的位置
        corners_coords = {
            "top_left": (0, 0, corner_size, corner_size),
            "top_right": (width - corner_size, 0, width, corner_size),
            "bottom_left": (0, height - corner_size, corner_size, height),
            "bottom_right": (width - corner_size, height - corner_size, width, height),
        }

        # 标记检测到折角的区域
        for corner_name in result.get("folded_corners", []):
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
        status_text = f"Fold: {result.get('has_fold', False)} (Conf: {result.get('confidence', 0):.2f})"
        cv2.putText(
            image,
            status_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if result.get("has_fold") else (255, 0, 0),
            2,
        )

        # 保存结果图片
        _, encoded_image = cv2.imencode(".jpg", image)
        encoded_image.tofile(output_path)

        return True

    except Exception as e:
        logging.error(f"生成可视化图片失败 {image_path}: {str(e)}")
        return False


def move_single_image(
    source_path: Path,
    target_dir: Path,
    result: dict,
    visualize: bool = False,
    move_lock: threading.Lock = None,
) -> bool:
    """
    移动单张图片到目标目录（可选生成可视化图片）

    Args:
        source_path: 源图片路径
        target_dir: 目标目录
        result: 检测结果
        visualize: 是否生成可视化图片
        move_lock: 线程锁（用于多线程安全）

    Returns:
        是否移动成功
    """
    try:
        # 确保目标目录存在
        if move_lock:
            with move_lock:
                target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        # 计算目标路径
        dest_path = target_dir / source_path.name

        # 如果目标文件已存在，添加序号
        if dest_path.exists():
            counter = 1
            stem = source_path.stem
            suffix = source_path.suffix
            while dest_path.exists():
                dest_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # 生成可视化图片（如果需要）
        if visualize:
            vis_stem = dest_path.stem
            vis_path = target_dir / f"{vis_stem}_marked{dest_path.suffix}"

            if visualize_fold_detection(str(source_path), result, str(vis_path)):
                logging.info(f"已生成可视化: {vis_path.name}")

        # 移动原图
        shutil.move(str(source_path), str(dest_path))
        logging.info(f"已移动: {source_path.name} -> {dest_path}")

        return True

    except Exception as e:
        logging.error(f"移动失败 {source_path.name}: {str(e)}")
        return False


def create_instant_move_callback(target_dir: str, visualize: bool = False):
    """
    创建即时移动的回调函数

    Args:
        target_dir: 目标目录
        visualize: 是否生成可视化图片

    Returns:
        回调函数
    """
    target_path = Path(target_dir)
    move_lock = threading.Lock()
    move_stats = {"success": 0, "failed": 0}
    stats_lock = threading.Lock()

    def callback(result: dict):
        """检测完成后的回调函数"""
        # 只处理检测到折角的图片
        if not result.get("has_fold", False):
            return

        source_path = Path(result["image_path"])

        # 移动图片
        success = move_single_image(
            source_path, target_path, result, visualize=visualize, move_lock=move_lock
        )

        # 更新统计
        with stats_lock:
            if success:
                move_stats["success"] += 1
            else:
                move_stats["failed"] += 1

    # 将统计信息附加到回调函数
    callback.stats = move_stats
    callback.stats_lock = stats_lock

    return callback


def print_results(results, stats, verbose=False):
    """打印检测结果"""
    print("\n" + "=" * 60)
    print("检测结果统计")
    print("=" * 60)
    print(f"总图片数:     {stats['total_images']}")
    print(f"检测到折角:   {stats['folded_images']}")
    print(f"正常图片:     {stats['normal_images']}")
    print(f"错误图片:     {stats['error_images']}")
    print(f"折角比例:     {stats['fold_rate']:.2%}")
    print("=" * 60)

    # 显示有折角的图片详情
    folded_images = [r for r in results if r["has_fold"]]
    if folded_images:
        print("\n检测到折角的图片:")
        print("-" * 60)
        for result in folded_images:
            path = Path(result["image_path"]).name
            corners = ", ".join(result["folded_corners"])
            confidence = result["confidence"]
            print(f"  • {path}")
            print(f"    折角位置: {corners}")
            print(f"    置信度: {confidence:.2f}")
            if verbose:
                print(f"    完整路径: {result['image_path']}")
        print("-" * 60)

    # 显示错误信息
    error_images = [r for r in results if r.get("error")]
    if error_images:
        print("\n处理错误的图片:")
        print("-" * 60)
        for result in error_images:
            path = Path(result["image_path"]).name
            error = result["error"]
            print(f"  • {path}: {error}")
            if verbose:
                print(f"    完整路径: {result['image_path']}")


def move_folded_images(results, target_dir):
    """
    将检测到折角的图片移动到目标目录

    Args:
        results: 检测结果列表
        target_dir: 目标目录路径

    Returns:
        移动成功的文件数量
    """
    target_path = Path(target_dir)

    # 创建目标目录（如果不存在）
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"创建目标目录失败: {str(e)}")
        return 0

    # 筛选出有折角的图片
    folded_images = [r for r in results if r["has_fold"]]

    if not folded_images:
        print("\n没有需要移动的图片。")
        return 0

    print(f"\n开始移动 {len(folded_images)} 张折角图片到: {target_path.absolute()}")

    moved_count = 0
    failed_count = 0

    for result in folded_images:
        source_path = Path(result["image_path"])
        dest_path = target_path / source_path.name

        # 如果目标文件已存在，添加序号
        if dest_path.exists():
            counter = 1
            stem = source_path.stem
            suffix = source_path.suffix
            while dest_path.exists():
                dest_path = target_path / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            shutil.move(str(source_path), str(dest_path))
            moved_count += 1
            logging.info(f"已移动: {source_path.name} -> {dest_path}")
        except Exception as e:
            failed_count += 1
            logging.error(f"移动失败 {source_path.name}: {str(e)}")

    print(f"\n移动完成: 成功 {moved_count} 张, 失败 {failed_count} 张")
    return moved_count


def save_results(results, stats, output_file):
    """保存结果到文件"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"图片折角检测报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

            # 统计信息
            f.write(f"统计信息:\n")
            f.write(f"  总图片数: {stats['total_images']}\n")
            f.write(f"  检测到折角: {stats['folded_images']}\n")
            f.write(f"  正常图片: {stats['normal_images']}\n")
            f.write(f"  错误图片: {stats['error_images']}\n")
            f.write(f"  折角比例: {stats['fold_rate']:.2%}\n\n")

            # 详细结果
            f.write(f"{'='*60}\n")
            f.write(f"详细检测结果:\n\n")

            for result in results:
                f.write(f"图片: {result['image_path']}\n")
                f.write(f"  有折角: {'是' if result['has_fold'] else '否'}\n")
                if result["has_fold"]:
                    f.write(f"  折角位置: {', '.join(result['folded_corners'])}\n")
                    f.write(f"  置信度: {result['confidence']:.2f}\n")
                if result.get("error"):
                    f.write(f"  错误: {result['error']}\n")
                f.write("\n")

        print(f"\n结果已保存到: {output_file}")

    except Exception as e:
        logging.error(f"保存结果文件失败: {str(e)}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 打印启动信息
    print("=" * 60)
    print("图片折角检测程序")
    print("=" * 60)
    print(f"检测目录: {args.directory}")
    print(f"线程数: {args.threads}")
    print(f"递归扫描: {'否' if args.no_recursive else '是'}")
    if args.target:
        print(f"目标目录: {args.target}")
        print(f"即时移动: {'是' if args.instant_move else '否'}")
        print(f"生成可视化: {'是' if args.visualize else '否'}")
    print("=" * 60)

    # 扫描图片文件
    scanner = ImageScanner()
    image_files = scanner.scan_directory(
        args.directory, recursive=not args.no_recursive
    )

    if not image_files:
        print("\n未找到图片文件！")
        sys.exit(0)

    print(f"\n找到 {len(image_files)} 张图片")

    # 创建检测器和线程池管理器
    detector = FoldDetector()
    thread_manager = ThreadPoolManager(num_threads=args.threads)

    # 准备即时移动的回调（如果需要）
    callback = None
    if args.target and args.instant_move:
        callback = create_instant_move_callback(args.target, visualize=args.visualize)
        print("\n即时移动模式已启用")

    # 执行检测
    print("\n开始检测...")
    results = thread_manager.process_images(
        image_files,
        detector,
        show_progress=not args.no_progress,
        on_result_callback=callback,
    )

    # 计算统计信息
    stats = thread_manager.get_statistics(results)

    # 显示结果
    print_results(results, stats, verbose=args.verbose)

    # 保存结果（如果指定了输出文件）
    if args.output:
        save_results(results, stats, args.output)

    # 处理移动（如果指定了目标目录且未使用即时移动）
    if args.target:
        if args.instant_move:
            # 显示即时移动的统计信息
            if callback:
                with callback.stats_lock:
                    moved = callback.stats["success"]
                    failed = callback.stats["failed"]
                print(f"\n移动完成: 成功 {moved} 张, 失败 {failed} 张")
        else:
            # 批量移动模式
            print("\n批量移动模式")
            # 如果需要可视化，先生成所有可视化图片
            if args.visualize:
                print("生成可视化图片...")
                target_path = Path(args.target)
                target_path.mkdir(parents=True, exist_ok=True)

                for result in results:
                    if result.get("has_fold", False):
                        source_path = Path(result["image_path"])
                        vis_path = (
                            target_path
                            / f"{source_path.stem}_marked{source_path.suffix}"
                        )
                        visualize_fold_detection(
                            str(source_path), result, str(vis_path)
                        )

            # 移动图片
            move_folded_images(results, args.target)

    print("\n检测完成！\n")


if __name__ == "__main__":
    main()
