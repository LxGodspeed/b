#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科学计算实验程序 - 适用于Linux环境
"""

import os
import sys
import argparse
from linux_env.compute_engine import ComputeEngine

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高性能科学计算实验")
    
    # 基本参数
    parser.add_argument("--cpu-only", action="store_true", help="仅使用CPU进行计算")
    parser.add_argument("--batch-size", type=int, default=200000, help="批处理大小，默认为200000")
    parser.add_argument("--data-file", type=str, default="utxo_data.pkl", help="数据文件路径")
    parser.add_argument("--token", type=str, help="通知服务令牌")
    
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
    args = parse_arguments()
    
    # 加载目标标识符（如果提供）
    target_identifiers = load_targets(args.target_file)
    
    # 创建计算引擎
    engine = ComputeEngine(
        target_identifiers=target_identifiers,
        data_file=args.data_file,
        use_gpu=not args.cpu_only,
        notification_token=args.token
    )
    
    # 设置批处理大小
    if args.batch_size:
        engine.set_batch_size(args.batch_size)
    
    try:
        # 运行实验
        print("\n开始科学计算实验，按Ctrl+C可停止程序")
        results = engine.run_experiment()
        
        # 简洁地总结结果
        if results:
            print(f"\n发现 {len(results)} 个有效结果")
        else:
            print("\n未发现任何有效结果")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
    finally:
        # 确保资源被正确释放
        if hasattr(engine, 'logger'):
            engine.logger.close()

if __name__ == "__main__":
    main() 