# 高性能科学计算实验框架

这是一个基于Python和CUDA的高性能科学计算实验框架，专为Linux环境设计，可利用GPU进行大规模并行计算。

## 功能特点

- 支持GPU加速计算，显著提升处理效率
- 灵活的批处理机制，自动优化为GPU特性
- 实时性能监控与报告
- 自动记录实验过程与发现
- 支持通知服务，及时推送重要发现
- 兼容CPU-only模式，适用于无GPU环境

## 安装依赖

```bash
# 基础依赖
pip install numpy hashlib base58 requests

# GPU加速依赖（如有NVIDIA GPU）
pip install cupy-cuda12x  # 根据CUDA版本选择对应的cupy包
pip install pynvml  # GPU监控工具（可选）
```

## 文件说明

- `data_loader.py`: 数据加载与处理模块
- `logger.py`: 实验日志记录模块
- `notifier.py`: 通知服务模块
- `compute_engine.py`: 核心计算引擎
- `run_experiment.py`: 主程序入口

## 使用方法

### 基本用法

```bash
# 使用默认配置运行
python -m linux_env.run_experiment

# 仅使用CPU运行
python -m linux_env.run_experiment --cpu-only

# 自定义批处理大小
python -m linux_env.run_experiment --batch-size 500000
```

### 高级选项

```bash
# 指定自定义数据文件
python -m linux_env.run_experiment --data-file path/to/data.pkl

# 使用目标识符文件
python -m linux_env.run_experiment --target-file targets.txt

# 启用通知服务
python -m linux_env.run_experiment --token your-notification-token
```

### 数据文件格式

- 数据文件为标准Python pickle格式
- 默认读取当前目录下的`utxo_data.pkl`文件
- 映射文件为可选，默认名为`address_map.pkl`

## 性能优化提示

1. 根据GPU型号适当调整批处理大小
2. 对于内存受限的环境，减小批处理大小
3. 使用`--cpu-only`可在无GPU环境运行

## 日志与结果

- 日志文件默认保存在`logs/`目录
- 每次运行会创建独立的日志文件，记录详细信息
- 重要发现将实时显示并记录

## 适用场景

- 大规模科学计算实验
- 密码学算法研究
- 数据库分析与验证
- 高性能计算教学演示

## 许可证

本项目仅用于科学研究与教育目的。请遵守当地法律法规。 