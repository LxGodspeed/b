#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能科学计算实验框架
=====================

这是一个用于进行高性能科学计算实验的框架，
支持GPU加速和大规模数据处理。
"""

__version__ = '1.0.0'
__author__ = 'Research Team'
__description__ = '高性能科学计算实验框架'

# 方便导入的模块列表
from linux_env.data_loader import DataLoader
from linux_env.logger import ExperimentLogger
from linux_env.notifier import ExperimentNotifier
from linux_env.compute_engine import ComputeEngine

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'status': 'stable'
} 