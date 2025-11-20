"""
定位点检测主程序
检测图片四角的定位标记点
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime

from config import (
    DEFAULT_THREAD_COUNT,
    LOG_LEVEL,
    LOG_FORMAT,
    DATE_FORMAT,
)
from marker_detector import MarkerDetector
from image_scanner import ImageScanner
from thread_pool import ThreadPoolManager


def setup_logging(verbose: bool = False):
    """配置日志系统"""
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL)
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="检测图片四角的定位标记点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python detect_markers.py -d ./images                    # 检测images目录下的所有图片
  python detect_markers.py -d ./images -t 8               # 使用8个线程
  python detect_markers.py -d ./images --target ./missing # 将缺失定位点的图片移动到missing目录
  python detect_markers.py -d ./images -o report.txt      # 保存结果报告
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
        "--target", type=str, help="将缺失定位点的图片移动到指定目录（可选）"
    )

    parser.add_argument(
        "--visualize", type=str, help="可视化检测结果并保存到指定目录（可选）"
    )

    return parser.parse_args()


def print_results(results, stats, verbose=False):
    """打印检测结果"""
    print("\n" + "=" * 70)
    print("定位点检测结果统计")
    print("=" * 70)
    print(f"总图片数:           {stats['total_images']}")
    print(
        f"完整定位点(4个):    {stats['complete_markers']} ({stats['complete_rate']:.1%})"
    )
    print(f"缺失定位点:         {stats['incomplete_markers']}")
    print(f"处理错误:           {stats['error_images']}")
    print("=" * 70)

    # 显示缺失定位点的图片详情
    incomplete_images = [
        r
        for r in results
        if not r.get("all_markers_found", False) and not r.get("error")
    ]
    if incomplete_images:
        print(f"\n缺失定位点的图片 ({len(incomplete_images)}):")
        print("-" * 70)
        for result in incomplete_images:
            path = Path(result["image_path"]).name
            found = len(result.get("found_markers", []))
            missing = ", ".join(result.get("missing_markers", []))
            print(f"  • {path}")
            print(f"    找到: {found}/4 个定位点")
            print(f"    缺失位置: {missing}")
            if verbose:
                print(f"    完整路径: {result['image_path']}")
        print("-" * 70)

    # 显示错误信息
    error_images = [r for r in results if r.get("error")]
    if error_images:
        print(f"\n处理错误的图片 ({len(error_images)}):")
        print("-" * 70)
        for result in error_images:
            path = Path(result["image_path"]).name
            error = result["error"]
            print(f"  • {path}: {error}")
        print("-" * 70)


def move_incomplete_images(results, target_dir):
    """
    将缺失定位点的图片移动到目标目录

    Args:
        results: 检测结果列表
        target_dir: 目标目录路径

    Returns:
        移动成功的文件数量
    """
    target_path = Path(target_dir)

    # 创建目标目录
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"创建目标目录失败: {str(e)}")
        return 0

    # 筛选出缺失定位点的图片
    incomplete_images = [
        r
        for r in results
        if not r.get("all_markers_found", False) and not r.get("error")
    ]

    if not incomplete_images:
        print("\n所有图片都有完整的定位点，无需移动。")
        return 0

    print(
        f"\n开始移动 {len(incomplete_images)} 张缺失定位点的图片到: {target_path.absolute()}"
    )

    moved_count = 0
    failed_count = 0

    for result in incomplete_images:
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
            f.write(f"定位点检测报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")

            # 统计信息
            f.write(f"统计信息:\n")
            f.write(f"  总图片数: {stats['total_images']}\n")
            f.write(f"  完整定位点(4个): {stats['complete_markers']}\n")
            f.write(f"  缺失定位点: {stats['incomplete_markers']}\n")
            f.write(f"  完整率: {stats['complete_rate']:.2%}\n")
            f.write(f"  错误图片: {stats['error_images']}\n\n")

            # 详细结果
            f.write(f"{'='*70}\n")
            f.write(f"详细检测结果:\n\n")

            for result in results:
                f.write(f"图片: {result['image_path']}\n")
                if result.get("error"):
                    f.write(f"  错误: {result['error']}\n")
                else:
                    f.write(
                        f"  完整定位点: {'是' if result.get('all_markers_found') else '否'}\n"
                    )
                    f.write(f"  找到: {len(result.get('found_markers', []))}/4 个\n")
                    if result.get("missing_markers"):
                        f.write(f"  缺失位置: {', '.join(result['missing_markers'])}\n")

                    # 详细的定位点信息
                    for corner, info in result.get("marker_details", {}).items():
                        if info.get("found"):
                            f.write(
                                f"    {corner}: 面积={info['area']}px, 得分={info['score']:.2f}\n"
                            )
                f.write("\n")

        print(f"\n结果已保存到: {output_file}")

    except Exception as e:
        logging.error(f"保存结果文件失败: {str(e)}")


def process_with_detector(detector, image_list, show_progress=True):
    """使用检测器处理图片列表"""
    from tqdm import tqdm

    results = []

    if show_progress:
        progress_bar = tqdm(total=len(image_list), desc="检测定位点", unit="image")

    for image_path in image_list:
        result = detector.detect_markers(image_path)
        results.append(result)

        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()

    return results


def calculate_statistics(results):
    """计算统计信息"""
    total = len(results)
    complete = sum(1 for r in results if r.get("all_markers_found", False))
    errors = sum(1 for r in results if r.get("error"))

    return {
        "total_images": total,
        "complete_markers": complete,
        "incomplete_markers": total - complete - errors,
        "error_images": errors,
        "complete_rate": complete / total if total > 0 else 0.0,
    }


def main():
    """主函数"""
    args = parse_arguments()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 打印启动信息
    print("=" * 70)
    print("图片定位点检测程序")
    print("=" * 70)
    print(f"检测目录: {args.directory}")
    print(f"线程数: {args.threads}")
    print(f"递归扫描: {'否' if args.no_recursive else '是'}")
    print("=" * 70)

    # 扫描图片文件
    scanner = ImageScanner()
    image_files = scanner.scan_directory(
        args.directory, recursive=not args.no_recursive
    )

    if not image_files:
        print("\n未找到图片文件！")
        sys.exit(0)

    print(f"\n找到 {len(image_files)} 张图片")

    # 创建检测器
    detector = MarkerDetector()

    # 执行检测
    print("\n开始检测定位点...")
    results = process_with_detector(
        detector, image_files, show_progress=not args.no_progress
    )

    # 计算统计信息
    stats = calculate_statistics(results)

    # 显示结果
    print_results(results, stats, verbose=args.verbose)

    # 保存结果
    if args.output:
        save_results(results, stats, args.output)

    # 移动缺失定位点的图片
    if args.target:
        move_incomplete_images(results, args.target)

    # 可视化（如果指定）
    if args.visualize:
        visualize_dir = Path(args.visualize)
        visualize_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n生成可视化结果到: {visualize_dir.absolute()}")

        # 只可视化缺失定位点的图片
        incomplete = [r for r in results if not r.get("all_markers_found", False)]
        for i, result in enumerate(incomplete[:10]):  # 限制前10张
            output_path = visualize_dir / f"visualize_{Path(result['image_path']).name}"
            detector.visualize_detection(result["image_path"], str(output_path))
            if i == 0:
                print(f"  示例: {output_path.name}")
        print(f"  共生成 {min(len(incomplete), 10)} 张可视化图片")

    print("\n检测完成！\n")


if __name__ == "__main__":
    main()
