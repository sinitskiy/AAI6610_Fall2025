#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Package
包含语义过滤和聚类分析模块
"""

# 导入核心类
from .semantic_filter import SemanticFilter
from .cluster_engine import ClusterEngine
from .visualizer import ClusterVisualizer

__all__ = [
    'SemanticFilter',
    'ClusterEngine',
    'ClusterVisualizer',
]

__version__ = '1.0.0'

# 默认配置
DEFAULT_CONFIG = {
    'semantic_filter': {
        'threshold': 30,
        'model': 'all-MiniLM-L6-v2',
        'batch_size': 32
    },
    'clustering': {
        'min_cluster_size': 10,
        'min_samples': 3,
        'methods': ['hdbscan', 'kmeans']
    }
}

def create_pipeline(config=None):
    """
    创建完整的聚类pipeline
    
    Args:
        config: 配置字典(可选)
        
    Returns:
        tuple: (filter, engine, visualizer)
    """
    config = config or DEFAULT_CONFIG
    
    filter_config = config.get('semantic_filter', DEFAULT_CONFIG['semantic_filter'])
    cluster_config = config.get('clustering', DEFAULT_CONFIG['clustering'])
    
    semantic_filter = SemanticFilter(
        threshold=filter_config['threshold'],
        model_name=filter_config['model']
    )
    
    cluster_engine = ClusterEngine(
        min_cluster_size=cluster_config['min_cluster_size'],
        min_samples=cluster_config['min_samples']
    )
    
    visualizer = ClusterVisualizer()
    
    return semantic_filter, cluster_engine, visualizer


print(f"🎯 Clustering package v{__version__} loaded")
