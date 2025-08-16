# OmniVision

**Unified Vision: Open-vocabulary segmentation meets representation learning and language grounding.**

*One framework for seeing, segmenting, and understanding everything.*

A powerful multi-modal AI pipeline combining **DINOv3**, **SAM 2**, and **CLIP** for advanced computer vision tasks, including image segmentation, video tracking, text-guided segmentation, and visual similarity computation.

## Features

- **Image Segmentation**: Click-based and text-guided object segmentation
- **Video Tracking**: Temporal object tracking across video frames
- **Text-to-Vision Bridge**: Natural language queries for visual understanding
- **Visual Search**: Cross-image similarity and correspondence finding
- **Interactive Demo**: Web-based Gradio interface for easy testing
- **Mac Optimized**: Native support for Apple Silicon (MPS)

## Quick Start

### Web Demo (Easiest)

```bash
# Launch interactive web demo
python launch_demo.py

# Or with custom settings
python launch_demo.py --host 0.0.0.0 --port 8080 --share
```

Then open http://localhost:7860 in your browser!

### CLI Usage

```bash
# Click-based segmentation
python -m omnivision.cli.main segment --image examples/images/cat.jpg --click 200,150

# Text-guided segmentation
python -m omnivision.cli.main text-segment --image examples/images/cat.jpg --text "a cat"

# Video tracking
python -m omnivision.cli.main track --video examples/videos/moving_circle.mp4 --click 100,200

# Text search
python -m omnivision.cli.main text-search --image examples/images/cat.jpg --queries "cat,dog,bird"
```

## Project Structure

```
omnivision/
├── omnivision/              # Main package
│   ├── models/              # Model wrappers (DINOv3, SAM2, CLIP)
│   ├── pipelines/           # Processing pipelines
│   ├── cli/                 # Command-line interface
│   ├── demos/               # Gradio web demo
│   └── utils/               # Utility functions
├── examples/                # Example data
│   ├── images/              # Test images
│   └── videos/              # Test videos
├── outputs/                 # Generated results
├── scripts/                 # Utility scripts
└── launch_demo.py           # Demo launcher
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)

### Setup

1. **Clone and setup environment:**
```bash
git clone https://github.com/DjKesu/OmniVision.git
cd OmniVision
conda env create -f environment.yml
conda activate dinov3-sam
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
conda install conda-forge::sam-2
```

3. **Test installation:**
```bash
python -m omnivision.cli.main test-model --verbose
```

## Key Capabilities

### 1. Click Segmentation
Point and click to segment any object with pixel-perfect precision.

### 2. Text-Guided Segmentation  
Use natural language like "the red car" or "person walking" to find and segment objects.

### 3. Video Object Tracking
Track objects across video frames with temporal consistency.

### 4. Visual Similarity Search
Find visual correspondences and compute similarity between images.

### 5. Multi-Query Search
Search images for multiple objects simultaneously and get relevance rankings.

## Architecture

### Core Models
- **DINOv3**: Self-supervised vision transformer for robust features
- **SAM 2**: Segment Anything Model 2 for precise segmentation  
- **CLIP**: Contrastive Language-Image Pre-training for text-vision bridge

### Pipeline Components
- **SegmentationPipeline**: DINOv3 + SAM 2 integration
- **VideoPipeline**: Temporal tracking and consistency
- **TextGuidedPipeline**: CLIP + DINOv3 + SAM 2 for language-guided segmentation
- **SimilarityPipeline**: Cross-image feature matching

## Performance

- **Speed**: ~0.5s per image on Apple M1/M2
- **Memory**: ~2-4GB for typical images
- **Accuracy**: State-of-the-art segmentation quality
- **Compatibility**: CPU, MPS (Mac), CUDA support

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `test-model` | Test model loading | `test-model --verbose` |
| `segment` | Click-based segmentation | `segment --image cat.jpg --click 200,150` |
| `text-segment` | Text-guided segmentation | `text-segment --image cat.jpg --text "cat"` |
| `text-search` | Multi-query search | `text-search --image cat.jpg --queries "cat,dog"` |
| `track` | Video object tracking | `track --video video.mp4 --click 100,200` |
| `similarity` | Image similarity | `similarity --img1 a.jpg --img2 b.jpg` |

## Examples

### Python API

```python
from omnivision import SegmentationPipeline, TextGuidedPipeline

# Click-based segmentation
seg_pipeline = SegmentationPipeline()
result = seg_pipeline.segment_by_click("image.jpg", (200, 150))

# Text-guided segmentation  
text_pipeline = TextGuidedPipeline()
result = text_pipeline.segment_by_text("image.jpg", "a red car")
```

### Advanced Usage

```python
# Multi-modal search
results = text_pipeline.search_and_segment(
    "image.jpg", 
    ["cat", "dog", "bird"], 
    top_k=2
)

# Video tracking
from omnivision import VideoPipeline
video_pipeline = VideoPipeline()
tracking_result = video_pipeline.track_by_similarity(
    frames, 
    reference_frame=0, 
    reference_coords=(100, 200)
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[DINOv3](https://github.com/facebookresearch/dinov3)** by Meta AI
- **[SAM 2](https://github.com/facebookresearch/segment-anything-2)** by Meta AI  
- **[CLIP](https://github.com/openai/CLIP)** by OpenAI
- Built with PyTorch, Transformers, and Gradio

## Citation

```bibtex
@software{omnivision,
  title={OmniVision: Unified Vision for Open-vocabulary Segmentation},
  author={OmniVision Contributors},
  year={2024},
  url={https://github.com/DjKesu/OmniVision}
}
```