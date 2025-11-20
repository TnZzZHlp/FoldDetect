# FoldDetect - 图片折角检测工具

一个支持多线程处理的Python图片折角检测工具，可以快速扫描指定目录下的图片并检测是否存在折角。

## 功能特点

- ✅ 支持多种图片格式（JPG, PNG, BMP, TIFF, WEBP等）
- ✅ 多线程并行处理，提高检测效率
- ✅ 基于OpenCV的智能折角检测算法
- ✅ 递归扫描目录及子目录
- ✅ 实时进度显示
- ✅ 详细的检测结果报告
- ✅ 可导出结果到文件

## 安装

本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理。

### 安装uv（如果尚未安装）

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装项目依赖

```bash
# 使用uv同步依赖
uv sync

# 或者使用uv pip安装
uv pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
# 检测指定目录下的所有图片
python main.py -d ./images

# 或使用uv运行
uv run main.py -d ./images
```

### 高级选项

```bash
# 使用8个线程进行处理
python main.py -d ./images -t 8

# 不递归扫描子目录
python main.py -d ./images --no-recursive

# 详细输出模式
python main.py -d ./images -v

# 保存结果到文件
python main.py -d ./images -o results.txt

# 组合使用多个选项
python main.py -d ./images -t 8 -v -o results.txt
```

### 命令行参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--directory` | `-d` | 要检测的图片目录路径（必需） | - |
| `--threads` | `-t` | 线程数 | CPU核心数 |
| `--no-recursive` | - | 不递归扫描子目录 | False |
| `--verbose` | `-v` | 详细输出模式 | False |
| `--no-progress` | - | 不显示进度条 | False |
| `--output` | `-o` | 输出结果文件路径 | - |

## 输出示例

```
============================================================
图片折角检测程序
============================================================
检测目录: ./test_images
线程数: 4
递归扫描: 是
============================================================

找到 15 张图片

开始检测...
Detecting folds: 100%|████████████████| 15/15 [00:02<00:00,  6.5image/s]

============================================================
检测结果统计
============================================================
总图片数:     15
检测到折角:   3
正常图片:     12
错误图片:     0
折角比例:     20.00%
============================================================

检测到折角的图片:
------------------------------------------------------------
  • photo1.jpg
    折角位置: top_right
    置信度: 0.30
  • photo2.jpg
    折角位置: bottom_left, bottom_right
    置信度: 0.60
  • scan001.png
    折角位置: top_left
    置信度: 0.30
------------------------------------------------------------

检测完成！
```

## 检测算法

程序使用基于OpenCV的多种图像处理技术进行折角检测：

1. **边缘检测** - Canny边缘检测算法
2. **角点检测** - Harris角点检测
3. **轮廓分析** - 检测异常几何形状
4. **区域分析** - 分别检测图片四个角落区域

## 支持的图片格式

- JPG / JPEG
- PNG
- BMP
- TIFF / TIF
- WEBP

## 项目结构

```
FoldDetect/
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── fold_detector.py     # 折角检测核心模块
├── image_scanner.py     # 图片扫描模块
├── thread_pool.py       # 多线程处理模块
├── requirements.txt     # 依赖列表
├── pyproject.toml       # 项目配置
└── README.md           # 本文档
```

## 依赖项

- Python >= 3.13
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- tqdm >= 4.66.0

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
