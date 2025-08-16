# DINOv3 Technical Implementation Details

## Architecture Deep Dive

### Vision Transformer Core
```
Input Shape: [B, 3, H, W]
Patch Size: 14 or 16 pixels
Embedding Dimension: 768 (ViT-B) / 1024 (ViT-L) / 1536 (ViT-G)
Number of Heads: 12 (ViT-B) / 16 (ViT-L) / 24 (ViT-G)
Number of Layers: 12 (ViT-B) / 24 (ViT-L) / 40 (ViT-G)
```

### Token Structure
```python
# Total tokens = 1 (CLS) + 4 (registers) + N (patches)
# For 224x224 image with patch_size=16:
# N = (224/16) * (224/16) = 14 * 14 = 196 patches
total_tokens = 1 + 4 + 196 = 201
```

### Register Tokens
- **Purpose**: Stabilize attention maps and prevent artifacts
- **Count**: 4 register tokens by default
- **Position**: Inserted after CLS token, before patch tokens
- **Training**: Learned during pre-training
- **Impact**: Cleaner attention patterns for dense prediction tasks

## Self-Supervised Training

### DINO Framework Components

#### Student-Teacher Architecture
```python
# Pseudo-code for DINO training
student = ViT(...)
teacher = ViT(...)  # EMA of student

for batch in dataloader:
    # Multi-crop augmentation
    global_crops = augment_global(batch)  # 2 views at 224x224
    local_crops = augment_local(batch)    # 8 views at 96x96
    
    # Student sees all views
    student_out = student(global_crops + local_crops)
    
    # Teacher sees only global views
    with torch.no_grad():
        teacher_out = teacher(global_crops)
    
    # Cross-entropy loss between student and teacher
    loss = cross_entropy(student_out, teacher_out.detach())
    
    # Update student
    loss.backward()
    optimizer.step()
    
    # Update teacher with EMA
    teacher_params = momentum * teacher_params + (1-momentum) * student_params
```

#### Loss Function
- **Primary Loss**: Cross-entropy between student and teacher outputs
- **Temperature Scaling**: Sharpening of probability distributions
- **Centering**: Prevents collapse to uniform predictions
- **No Negative Pairs**: Unlike contrastive methods

### Training Details

#### Data Augmentation
- **Global Crops**: 2 views at 224x224 resolution
- **Local Crops**: 8 views at 96x96 resolution
- **Color Jittering**: Standard ImageNet augmentations
- **Random Flipping**: Horizontal flips only
- **Gaussian Blur**: Applied probabilistically

#### Optimization
```yaml
optimizer: AdamW
learning_rate: 0.0005
weight_decay: 0.04
batch_size: 4096
warmup_epochs: 10
total_epochs: 100
momentum_teacher: 0.996 -> 1.0 (cosine schedule)
```

## Feature Extraction Pipeline

### Preprocessing
```python
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vitb16-pretrain-lvd1689m"
)

# Default preprocessing
processed = processor(
    images=image,
    return_tensors="pt",
    do_resize=True,
    size={"height": 518, "width": 518},  # DINOv3 default
    do_center_crop=True,
    crop_size={"height": 518, "width": 518},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)
```

### Token Extraction
```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

with torch.no_grad():
    outputs = model(**processed)
    
# Extract different token types
last_hidden = outputs.last_hidden_state  # [B, num_tokens, dim]
cls_token = last_hidden[:, 0, :]        # [B, dim]
register_tokens = last_hidden[:, 1:5, :] # [B, 4, dim]
patch_tokens = last_hidden[:, 5:, :]    # [B, H*W, dim]

# Reshape patches to grid
H = W = int(patch_tokens.shape[1] ** 0.5)
patch_grid = patch_tokens.reshape(B, H, W, -1)
```

## Quantization Support

### Int4 Weight-Only Quantization
```python
from transformers import AutoModel
from torchao.quantization import int4_weight_only

# Load model with int4 quantization
model = AutoModel.from_pretrained(
    "facebook/dinov3-vitg-7b",
    torch_dtype=torch.bfloat16,
    quantization_config=int4_weight_only()
)

# Memory comparison
# FP16: ~14GB VRAM for ViT-G
# Int4: ~3.5GB VRAM for ViT-G
```

### Performance Trade-offs
| Model | Precision | Memory | Speed | Accuracy |
|-------|-----------|--------|-------|----------|
| ViT-B | FP32 | 350MB | 1.0x | 100% |
| ViT-B | FP16 | 175MB | 1.8x | 99.9% |
| ViT-B | Int8 | 90MB | 2.5x | 99.5% |
| ViT-G | FP32 | 28GB | 1.0x | 100% |
| ViT-G | FP16 | 14GB | 1.8x | 99.9% |
| ViT-G | Int4 | 3.5GB | 3.2x | 98.5% |

## Dense Prediction Adaptations

### Object Detection Head
```python
# Pseudo-code for detection adaptation
class DINOv3Detector:
    def __init__(self, backbone):
        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork()
        self.head = DetectionHead()
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone.get_intermediate_layers(x)
        
        # Build feature pyramid
        pyramid = self.fpn(features)
        
        # Predict boxes and classes
        detections = self.head(pyramid)
        return detections
```

### Semantic Segmentation
```python
# Upsampling patch features for segmentation
def segment_with_dinov3(model, image, num_classes):
    # Get patch features
    features = model(image)
    patch_tokens = features[:, 5:, :]  # Skip CLS and registers
    
    # Reshape to spatial grid
    B, N, D = patch_tokens.shape
    H = W = int(N ** 0.5)
    spatial_features = patch_tokens.reshape(B, H, W, D)
    
    # Upsample to original resolution
    spatial_features = spatial_features.permute(0, 3, 1, 2)
    upsampled = F.interpolate(
        spatial_features,
        size=image.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    
    # Linear projection to classes
    segmentation = linear_head(upsampled)
    return segmentation
```

## Video Processing

### Temporal Consistency
```python
class DINOv3VideoProcessor:
    def __init__(self, model):
        self.model = model
        self.memory_bank = []
    
    def process_frame(self, frame, use_memory=True):
        # Extract features
        features = self.model(frame)
        
        if use_memory and self.memory_bank:
            # Temporal smoothing with previous frames
            smoothed = 0.7 * features + 0.3 * self.memory_bank[-1]
            features = smoothed
        
        # Update memory
        self.memory_bank.append(features)
        if len(self.memory_bank) > 5:
            self.memory_bank.pop(0)
        
        return features
```

## Optimization Techniques

### Compilation
```python
import torch

# Compile model for faster inference
model = torch.compile(
    model,
    mode="reduce-overhead",
    fullgraph=True
)
```

### Batching Strategies
```python
# Efficient batch processing
def process_batch_efficiently(model, images, batch_size=32):
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = model(batch)
        
        results.append(features)
    
    return torch.cat(results, dim=0)
```

## Memory Management

### VRAM Requirements
| Model | Batch Size | Resolution | FP32 | FP16 | Int8 | Int4 |
|-------|------------|------------|------|------|------|------|
| ViT-B | 1 | 518x518 | 1.2GB | 0.6GB | 0.4GB | 0.3GB |
| ViT-B | 32 | 518x518 | 8GB | 4GB | 2.5GB | 1.5GB |
| ViT-L | 1 | 518x518 | 3GB | 1.5GB | 0.9GB | 0.6GB |
| ViT-L | 32 | 518x518 | 20GB | 10GB | 6GB | 3.5GB |
| ViT-G | 1 | 518x518 | 30GB | 15GB | 8GB | 4GB |
| ViT-G | 8 | 518x518 | OOM | 40GB | 22GB | 12GB |

### Optimization Tips
1. **Use Mixed Precision**: FP16/BF16 for 2x memory savings
2. **Gradient Checkpointing**: For training with limited VRAM
3. **CPU Offloading**: Move unused layers to CPU
4. **Quantization**: Int8/Int4 for edge deployment
5. **Batch Size Tuning**: Find optimal batch size for hardware