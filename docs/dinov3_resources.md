# DINOv3 Resources and References

## Official Resources

### Meta AI Resources
- **Blog Post**: [DINOv3: Self-supervised learning for vision at unprecedented scale](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
- **Research Page**: [DINOv3 Research - AI at Meta](https://ai.meta.com/research/publications/dinov3/)
- **GitHub Repository**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

### Hugging Face Models
- **ViT-B Model**: [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- **ViT-L Model**: [facebook/dinov3-vitl14-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl14-pretrain-lvd1689m)
- **ViT-G Model**: [facebook/dinov3-vitg-7b](https://huggingface.co/facebook/dinov3-vitg-7b)
- **Documentation**: [Transformers DINOv3 Docs](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3)

## Related Research Papers

### Core DINOv3 Papers
- **Vision Transformers Need Registers** (2023)
  - [arXiv](https://arxiv.org/html/2309.16588v2)
  - [OpenReview](https://openreview.net/forum?id=2dnO3LLiJ1)
  - Key insight: Register tokens stabilize attention maps

### DINO Family Papers
- **DINO: Emerging Properties in Self-Supervised Vision Transformers** (2021)
  - [arXiv](https://arxiv.org/abs/2104.14294)
  - Original DINO paper introducing the self-supervised method

- **DINOv2: Learning Robust Visual Features without Supervision** (2023)
  - [arXiv](https://arxiv.org/abs/2304.07193)
  - Scaling up DINO with larger models and datasets

## Datasets for Evaluation

### Visual Correspondence
- **SPair-71k**: [Dataset Page](https://cvlab.postech.ac.kr/research/SPair-71k/)
  - 70,958 paired images with semantic correspondences
  - Strong viewpoint and scale changes

- **HPatches**: [GitHub](https://github.com/hpatches/hpatches-dataset)
  - Local patch matching benchmark
  - Illumination and viewpoint variations

### Video Segmentation
- **DAVIS 2017**: [Official Site](https://davischallenge.org/)
  - Semi-supervised video object segmentation
  - Standard benchmark for video tracking

- **YouTube-VOS**: [Dataset Page](https://youtube-vos.org/dataset/)
  - Large-scale video object segmentation
  - Multiple evaluation protocols

- **OVIS**: [Project Page](https://songbai.site/ovis/)
  - Occluded video instance segmentation
  - Challenging occlusion scenarios

### Referring Expression
- **RefCOCO/RefCOCO+/RefCOCOg**: [GitHub API](https://github.com/lichengunc/refer)
  - Referring expression comprehension
  - Natural language grounding

- **Flickr30k Entities**: [GitHub](https://github.com/BryanPlummer/flickr30k_entities)
  - Phrase-to-region annotations
  - Clean supervision for grounding

- **Localized Narratives**: [Google Research](https://google.github.io/localized-narratives/)
  - 849k images with word-level grounding
  - Mouse-trace annotations

### Large Vocabulary
- **LVIS**: [Hugging Face Dataset](https://huggingface.co/datasets/Voxel51/LVIS)
  - Large vocabulary instance segmentation
  - 1000+ object categories

## Integration Libraries

### SAM 2 (Segment Anything Model 2)
- **GitHub**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **License**: Apache 2.0
- **Features**: Video tracking, streaming memory, prompt-based segmentation

### 3D Gaussian Splatting
- **Original 3DGS**: [Paper](https://arxiv.org/abs/2308.04079)
- **Feature 3DGS**: [Project Page](https://feature-3dgs.github.io/)
- **LangSplat**: [GitHub](https://github.com/minghanqin/LangSplat)
  - Language-guided 3D editing

## Community Resources

### Tutorials and Guides
- **Building DINO from Scratch**: [Medium Article](https://medium.com/thedeephub/self-supervised-vision-transformer-implementing-the-dino-model-from-scratch-with-pytorch-62203911bcc9)
- **DINOv2 Complete Guide**: [Medium Article](https://medium.com/data-science-in-your-pocket/dinov2-a-complete-guide-to-self-supervised-learning-and-vision-transformers-d5c1fb75d93f)
- **DINOv2 by Meta**: [LearnOpenCV Tutorial](https://learnopencv.com/dinov2-self-supervised-vision-transformer/)

### News and Coverage
- **Meta AI Unveils DINOv3**: [MarkTechPost](https://www.marktechpost.com/2025/08/14/meta-ai-just-released-dinov3-a-state-of-the-art-computer-vision-model-trained-with-self-supervised-learning-generating-high-resolution-image-features/)
- **DINOv3 Breakthrough**: [ZoomHoot](https://zoomhoot.com/2025/08/14/meta-ai-unveils-dinov3-a-cutting-edge-self-supervised-computer-vision-model/)
- **Investment News**: [Investing.com](https://www.investing.com/news/stock-market-news/meta-introduces-dinov3-a-breakthrough-in-selfsupervised-vision-ai-93CH-4193181)

## Code Examples and Implementations

### Official Implementations
- **DINOv3 Training Code**: Available in official repo (gated access)
- **Evaluation Scripts**: Included with model releases
- **Downstream Adapters**: Sample notebooks on Hugging Face

### Community Implementations
- **PyTorch DINO**: [facebookresearch/dino](https://github.com/facebookresearch/dino)
- **DINOv2 PyTorch**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

## Licensing Information

### DINOv3 License
- **Type**: Commercial license with gating
- **Access**: Via Hugging Face model hub
- **Usage**: Commercial use permitted with license acceptance
- **Requirements**: Must accept terms on Hugging Face

### Alternative Licenses
- **DINOv2**: Apache 2.0 (fully open source)
- **DINO (v1)**: Apache 2.0
- **SAM 2**: Apache 2.0

## Performance Benchmarks

### Model Comparison Table
| Model | Parameters | Training Data | License | DAVIS J&F | SPair PCK |
|-------|------------|---------------|---------|-----------|-----------|
| DINO | 85M | 1.3M images | Apache 2.0 | - | 42.1 |
| DINOv2-B | 86M | 142M images | Apache 2.0 | 76.2 | 65.3 |
| DINOv2-L | 300M | 142M images | Apache 2.0 | 78.5 | 71.2 |
| DINOv2-G | 1.1B | 142M images | Apache 2.0 | 81.3 | 75.8 |
| DINOv3-B | 86M | 1.7B images | Commercial | 82.1 | 78.4 |
| DINOv3-L | 300M | 1.7B images | Commercial | 84.7 | 82.1 |
| DINOv3-G | 7B | 1.7B images | Commercial | 87.3 | 85.6 |

### Hardware Requirements
| Model | VRAM (FP32) | VRAM (FP16) | VRAM (Int4) | Inference Speed |
|-------|-------------|-------------|-------------|-----------------|
| ViT-B | 350MB | 175MB | 90MB | 45ms/image |
| ViT-L | 1.2GB | 600MB | 300MB | 85ms/image |
| ViT-G | 28GB | 14GB | 3.5GB | 320ms/image |

## Tools and Utilities

### Quantization
- **TorchAO**: Int4 weight-only quantization
- **Documentation**: [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3#quantization)

### Visualization
- **Attention Maps**: Tools for visualizing self-attention
- **Feature Maps**: Patch token visualization utilities
- **Correspondence**: Matching visualization tools

### Deployment
- **ONNX Export**: Supported via Transformers
- **TorchScript**: Compilation for production
- **Edge Deployment**: Quantized models for mobile/edge

## Future Developments

### Upcoming Features
- Enhanced 3D integration capabilities
- Improved video processing pipelines
- Better language-vision alignment
- Efficient deployment strategies

### Research Directions
- Multi-modal fusion with DINOv3
- Zero-shot capabilities enhancement
- Continual learning approaches
- Domain adaptation methods

## Getting Help

### Official Channels
- **GitHub Issues**: Report bugs and feature requests
- **Hugging Face Discussions**: Model-specific questions
- **Meta AI Forums**: Research discussions

### Community Support
- **Discord Servers**: Computer vision communities
- **Reddit**: r/MachineLearning, r/computervision
- **Stack Overflow**: Technical implementation questions

## Citation

If you use DINOv3 in your research, please cite:
```bibtex
@article{dinov3_2025,
  title={DINOv3: Self-supervised learning for vision at unprecedented scale},
  author={Meta AI Research},
  journal={Meta AI Blog},
  year={2025},
  month={August}
}
```