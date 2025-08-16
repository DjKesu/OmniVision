"""Pipeline implementations for OmniVision"""

from .similarity import SimilarityPipeline
from .localization import LocalizationPipeline

# Import optional pipelines
pipelines = ["SimilarityPipeline", "LocalizationPipeline"]

try:
    from .segmentation import SegmentationPipeline
    from .video import VideoPipeline
    pipelines.extend(["SegmentationPipeline", "VideoPipeline"])
except ImportError:
    pass

try:
    from .text_guided import TextGuidedPipeline
    pipelines.append("TextGuidedPipeline")
except ImportError:
    pass

__all__ = pipelines