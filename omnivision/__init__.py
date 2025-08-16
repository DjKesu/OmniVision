"""
OmniVision: Unified Vision for Open-vocabulary Segmentation

Unified Vision: Open-vocabulary segmentation meets representation learning and language grounding.
One framework for seeing, segmenting, and understanding everything.
"""

__version__ = "0.1.0"
__author__ = "OmniVision Contributors"

from .models import DINOv3Backbone
from .pipelines import SimilarityPipeline, LocalizationPipeline

__all__ = [
    "DINOv3Backbone",
    "SimilarityPipeline", 
    "LocalizationPipeline"
]