"""
Benchmark module for OmniVision evaluation

This module provides comprehensive evaluation capabilities for text-guided
segmentation models on standard benchmark datasets.
"""

from .datasets import *
from .metrics import *
from .evaluator import *

__all__ = [
    'RefCOCODataset',
    'RefCOCOPlusDataset', 
    'RefCOCOgDataset',
    'SegmentationMetrics',
    'BenchmarkEvaluator'
]