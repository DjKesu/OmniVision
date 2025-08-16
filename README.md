# OmniVision

**Unified Vision: Open-vocabulary segmentation meets representation learning and language grounding.**

*One framework for seeing, segmenting, and understanding everything.*

A powerful multi-modal AI pipeline combining **DINOv3**, **SAM 2**, and **CLIP** for advanced computer vision tasks, including image segmentation, video tracking, text-guided segmentation, and visual similarity computation.

## Features

- **Visual Similarity**: Cross-image feature matching and correspondence analysis
- **Object Localization**: Click-based object detection and localization
- **Multi-Modal Fusion**: Advanced fusion architectures for combining vision models
- **Flexible Backends**: Support for DINOv3, CLIP, and SAM2 (optional)
- **Interactive Demo**: Web-based Gradio interface for easy testing
- **Mac Optimized**: Native support for Apple Silicon (MPS)
- **Extensible**: Modular pipeline architecture for custom workflows

## Quick Start

### Web Demo (Easiest)

```bash
# Launch interactive web demo
python -m omnivision.demos.gradio_app

# Or run the Gradio app directly
python omnivision/demos/gradio_app.py
```

Then open the provided URL in your browser!

### CLI Usage

```bash
# Test model loading
python -m omnivision.cli.main test-model --verbose

# Compute similarity between images
python -m omnivision.cli.main similarity --ref examples/images/cat.jpg --target examples/images/dog.jpg

# Localization and correspondence
python -m omnivision.cli.main localize --ref examples/images/cat.jpg --target examples/images/cat.jpg --click 200,150

# Available commands: test-model, similarity, localize
```

## Project Structure

```
omnivision/
├── omnivision/              # Main package
│   ├── models/              # Model wrappers and fusion modules
│   │   ├── dinov3_backbone.py
│   │   ├── clip_wrapper.py
│   │   ├── sam2_wrapper.py
│   │   ├── cross_modal_fusion.py
│   │   ├── improved_fusion.py
│   │   ├── simple_fusion.py
│   │   └── trident_fusion.py
│   ├── pipelines/           # Processing pipelines
│   │   ├── similarity.py
│   │   ├── localization.py
│   │   ├── segmentation.py
│   │   ├── text_guided.py
│   │   └── video.py
│   ├── cli/                 # Command-line interface
│   ├── demos/               # Gradio web demo
│   ├── utils/               # Utility functions
│   ├── data/                # Data handling
│   ├── configs/             # Configuration files
│   └── tests/               # Unit tests
├── benchmarks/              # Evaluation and benchmarking
├── examples/                # Example data and test images/videos
├── outputs/                 # Generated results and experiments
├── scripts/                 # Utility scripts
└── tests/                   # Integration tests and results
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
# Optional: Install SAM2 for segmentation capabilities
# conda install conda-forge::sam-2
```

3. **Test installation:**
```bash
python -m omnivision.cli.main test-model --verbose
```

## Key Capabilities

### 1. Visual Similarity and Correspondence
Compute dense feature similarities between images using DINOv3 embeddings and find corresponding regions across different images.

### 2. Object Localization
Click-based object detection that finds similar objects or regions in target images based on reference patches.

### 3. Multi-Modal Fusion
Advanced fusion architectures (Trident, Cross-Modal, Improved) that combine different vision models for enhanced performance.

### 4. Flexible Model Integration
Modular design supporting DINOv3 as core backbone with optional CLIP and SAM2 integration for extended capabilities.

### 5. Research-Ready Framework
Comprehensive benchmarking tools, evaluation metrics, and extensible pipeline architecture for computer vision research.

## Architecture

### Core Models
- **DINOv3Backbone**: Self-supervised vision transformer for robust features
- **CLIPWrapper**: Text-image understanding and bridging
- **SAM2Wrapper**: Segment Anything Model 2 for precise segmentation

### Fusion Modules
- **CrossModalFusion**: Multi-modal feature integration
- **TridentFusion**: Advanced three-way fusion architecture
- **ImprovedFusion**: Enhanced fusion with attention mechanisms
- **SimpleFusion**: Lightweight fusion approach

### Pipeline Components
- **SimilarityPipeline**: Cross-image feature matching and correspondence
- **LocalizationPipeline**: Object localization and detection
- **SegmentationPipeline**: DINOv3 + SAM 2 integration (optional)
- **TextGuidedPipeline**: CLIP + DINOv3 + SAM 2 for language-guided tasks (optional)
- **VideoPipeline**: Temporal tracking and consistency (optional)

## Performance

- **Speed**: ~0.5s per image on Apple M1/M2
- **Memory**: ~2-4GB for typical images
- **Accuracy**: State-of-the-art segmentation quality
- **Compatibility**: CPU, MPS (Mac), CUDA support

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `test-model` | Test model loading and functionality | `test-model --model base --verbose` |
| `similarity` | Compute image similarity and correspondence | `similarity --ref cat.jpg --target dog.jpg` |
| `localize` | Object localization with click input | `localize --ref cat.jpg --target scene.jpg --click 200,150` |

## Examples

### Python API

```python
from omnivision import DINOv3Backbone, SimilarityPipeline, LocalizationPipeline

# Initialize backbone
backbone = DINOv3Backbone()

# Compute image similarities
similarity_pipeline = SimilarityPipeline(backbone)
similarity_map = similarity_pipeline.compute_similarity("ref.jpg", "target.jpg")

# Object localization
localization_pipeline = LocalizationPipeline(backbone)
result = localization_pipeline.localize_by_click("ref.jpg", "target.jpg", (200, 150))
```

### Advanced Usage

```python
# Multi-modal fusion (if available)
from omnivision.models.trident_fusion import TridentFusion
from omnivision.pipelines.text_guided import TextGuidedPipeline

# Advanced fusion architecture
fusion_model = TridentFusion()
text_pipeline = TextGuidedPipeline(fusion_model)

# Cross-modal understanding
result = text_pipeline.search_by_text("image.jpg", "a red car")
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