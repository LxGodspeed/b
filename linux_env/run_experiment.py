#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科学计算实验程序 - 适用于Linux环境
"""

import os
import sys
import argparse

# 添加当前目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入计算引擎，不使用linux_env前缀
from compute_engine import ComputeEngine

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高性能科学计算实验")
    
    # 基本参数
    parser.add_argument("--cpu-only", action="store_true", help="仅使用CPU进行计算")
    parser.add_argument("--batch-size", type=int, default=500000, help="批处理大小，默认为500000")
    parser.add_argument("--data-file", type=str, default="utxo_data.pkl", help="数据文件路径")
    parser.add_argument("--token", type=str, help="通知服务令牌")
    parser.add_argument("--force-gpu", action="store_true", help="强制使用GPU（忽略兼容性检查）")
    parser.add_argument("--debug", action="store_true", help="调试模式（更多日志输出）")
    
    # 目标文件
    parser.add_argument("--target-file", type=str, help="包含目标标识符的文件")
    
    return parser.parse_args()

def load_targets(file_path):
    """从文件加载目标标识符"""
    if not file_path or not os.path.exists(file_path):
        return []
        
    targets = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                identifier = line.strip()
                if identifier:
                    targets.append(identifier)
        print(f"已加载 {len(targets)} 个目标标识符")
    except Exception as e:
        print(f"加载目标文件出错: {e}")
    
    return targets

def main():
    """主函数"""
    # 确保所有输出都能正确处理Unicode字符
    try:
        # 再次设置默认编码，以防万一
        import locale
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass
    
    args = parse_arguments()
    
    # 默认使用CPU模式运行，不需要--cpu-only参数
    use_cpu_mode = True
    if args.force_gpu:
        use_cpu_mode = False  # 只有明确指定--force-gpu时才使用GPU
        
    # 在调试模式下设置环境变量
    if args.debug:
        # 强制禁用输出缓冲
        os.environ['PYTHONUNBUFFERED'] = '1'
        print("调试模式已启用", flush=True)
    
    # 根据命令行参数设置环境变量
    if not args.force_gpu:
        # 检查CuPy版本与CUDA版本是否匹配
        try:
            import pkg_resources
            cupy_package = None
            for pkg in pkg_resources.working_set:
                if pkg.key.startswith('cupy-cuda'):
                    cupy_package = pkg.key
                    break
            
            if cupy_package == 'cupy-cuda12x' and os.environ.get('CUDA_HOME', '').endswith('cuda-11.7'):
                print("\n警告: 检测到CUDA版本不匹配!", flush=True)
                print("系统CUDA版本: 11.7", flush=True)
                print(f"安装的CuPy包: {cupy_package}", flush=True)
                print("\n需要解决版本不匹配问题，请选择以下方案之一:", flush=True)
                print("1. 安装匹配版本: pip uninstall cupy-cuda12x && pip install cupy-cuda11x", flush=True)
                print("2. 使用CPU模式: python run_experiment.py --cpu-only", flush=True)
                print("3. 强制使用GPU: python run_experiment.py --force-gpu (不推荐，可能不稳定)", flush=True)
                
                use_cpu_mode = True
        except:
            pass
    
    # 加载目标标识符（如果提供）
    target_identifiers = load_targets(args.target_file)
    
    # 创建计算引擎
    engine = ComputeEngine(
        target_identifiers=target_identifiers,
        data_file=args.data_file,
        use_gpu=not use_cpu_mode,
        notification_token=args.token
    )
    
    # 设置批处理大小
    if args.batch_size:
        engine.set_batch_size(args.batch_size)
    
    # 如果是CPU模式，调整批处理大小
    if use_cpu_mode and args.batch_size == 500000:
        # 减小CPU模式下的默认批处理大小
        smaller_batch = 50000
        print(f"CPU模式下自动调整批处理大小: {smaller_batch}", flush=True)
        engine.set_batch_size(smaller_batch)
    
    try:
        # 运行实验
        print("\n开始科学计算实验，按Ctrl+C可停止程序", flush=True)
        # 显示运行模式信息
        if use_cpu_mode:
            print("运行模式: CPU", flush=True)
        else:
            print("运行模式: GPU", flush=True)
        
        results = engine.run_experiment()
        
        # 简洁地总结结果
        if results:
            print(f"\n发现 {len(results)} 个有效结果", flush=True)
        else:
            print("\n未发现任何有效结果", flush=True)
    except KeyboardInterrupt:
        print("\n程序被用户中断", flush=True)
    except Exception as e:
        # 错误信息可能包含非ASCII字符，使用repr确保可打印
        try:
            if args.debug:
                import traceback
                traceback.print_exc()
            else:
                print(f"\n程序出错: {repr(e)}", flush=True)
        except:
            print("\n程序运行时出现错误", flush=True)
    finally:
        # 确保资源被正确释放
        if hasattr(engine, 'logger'):
            try:
                engine.logger.close()
            except:
                pass

if __name__ == "__main__":
    main() 