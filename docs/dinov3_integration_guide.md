# DINOv3 Integration Guide

## Getting Started

### Installation
```bash
# Install transformers with vision support
pip install transformers>=4.40.0
pip install torch torchvision
pip install pillow

# Optional: for quantization
pip install torchao

# Optional: for video processing
pip install opencv-python
```

### Basic Usage

#### Loading the Model
```python
from transformers import AutoModel, AutoImageProcessor
import torch
from PIL import Image

# Choose model variant
model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # ViT-B
# model_name = "facebook/dinov3-vitl14-pretrain-lvd1689m"  # ViT-L
# model_name = "facebook/dinov3-vitg-7b"  # ViT-G (7B params)

# Load model and processor
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoImageProcessor.from_pretrained(model_name)

# Process image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Extract features
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state
```

#### Extracting Different Token Types
```python
def extract_dinov3_features(model, processor, image):
    """Extract CLS, register, and patch tokens from DINOv3"""
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get all tokens
    last_hidden = outputs.last_hidden_state  # [B, N_tokens, D]
    
    # Split tokens
    num_registers = model.config.num_register_tokens  # 4
    
    cls_token = last_hidden[:, 0, :]  # [B, D]
    register_tokens = last_hidden[:, 1:1+num_registers, :]  # [B, 4, D]
    patch_tokens = last_hidden[:, 1+num_registers:, :]  # [B, H*W, D]
    
    # Reshape patches to grid
    B, N, D = patch_tokens.shape
    H = W = int(N ** 0.5)
    patch_grid = patch_tokens.reshape(B, H, W, D)
    
    return {
        'cls': cls_token,
        'registers': register_tokens,
        'patches': patch_tokens,
        'patch_grid': patch_grid
    }
```

## Common Use Cases

### 1. Image Similarity and Retrieval

```python
import torch.nn.functional as F
import numpy as np

class DINOv3ImageRetrieval:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.database = []
        
    def add_to_database(self, images, metadata=None):
        """Add images to retrieval database"""
        features = []
        
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token for global representation
                cls_features = outputs.last_hidden_state[:, 0, :]
                cls_features = F.normalize(cls_features, dim=-1)
                features.append(cls_features.cpu())
        
        self.database.extend(zip(features, metadata or [None]*len(images)))
    
    def search(self, query_image, top_k=5):
        """Find similar images in database"""
        # Extract query features
        inputs = self.processor(images=query_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_features = outputs.last_hidden_state[:, 0, :]
            query_features = F.normalize(query_features, dim=-1)
        
        # Compute similarities
        similarities = []
        for db_features, metadata in self.database:
            sim = (query_features @ db_features.T).item()
            similarities.append((sim, metadata))
        
        # Return top-k
        similarities.sort(reverse=True)
        return similarities[:top_k]
```

### 2. Dense Correspondence Matching

```python
def find_correspondences(model, processor, img1, img2, num_matches=10):
    """Find corresponding points between two images"""
    
    # Extract patch features for both images
    features1 = extract_dinov3_features(model, processor, img1)
    features2 = extract_dinov3_features(model, processor, img2)
    
    # Get patch grids
    patches1 = features1['patch_grid'][0]  # [H, W, D]
    patches2 = features2['patch_grid'][0]  # [H, W, D]
    
    # Normalize
    patches1 = F.normalize(patches1.reshape(-1, patches1.shape[-1]), dim=-1)
    patches2 = F.normalize(patches2.reshape(-1, patches2.shape[-1]), dim=-1)
    
    # Compute similarity matrix
    similarity = patches1 @ patches2.T  # [HW1, HW2]
    
    # Find best matches
    matches = []
    for _ in range(num_matches):
        # Find maximum similarity
        max_idx = similarity.argmax()
        idx1 = max_idx // similarity.shape[1]
        idx2 = max_idx % similarity.shape[1]
        
        # Convert to coordinates
        H1, W1 = features1['patch_grid'].shape[1:3]
        H2, W2 = features2['patch_grid'].shape[1:3]
        
        y1, x1 = idx1 // W1, idx1 % W1
        y2, x2 = idx2 // W2, idx2 % W2
        
        matches.append(((x1.item(), y1.item()), (x2.item(), y2.item())))
        
        # Suppress this match
        similarity[idx1, :] = -1
        similarity[:, idx2] = -1
    
    return matches
```

### 3. Semantic Segmentation Adapter

```python
class DINOv3Segmenter:
    def __init__(self, model, processor, num_classes):
        self.model = model
        self.processor = processor
        self.num_classes = num_classes
        
        # Simple linear head for segmentation
        self.seg_head = torch.nn.Conv2d(
            model.config.hidden_size,
            num_classes,
            kernel_size=1
        ).to(model.device)
    
    def segment(self, image):
        """Perform semantic segmentation"""
        # Extract features
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get patch tokens
        num_registers = self.model.config.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, 1+num_registers:, :]
        
        # Reshape to spatial
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        spatial_features = patch_tokens.reshape(B, H, W, D)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # Apply segmentation head
        logits = self.seg_head(spatial_features)
        
        # Upsample to original resolution
        H_orig, W_orig = image.size[::-1]  # PIL image
        segmentation = F.interpolate(
            logits,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )
        
        return segmentation.argmax(dim=1)
```

### 4. Object Detection with Patches

```python
def detect_objects_by_similarity(model, processor, image, reference_patch, threshold=0.7):
    """Detect objects similar to a reference patch"""
    
    # Extract features
    features = extract_dinov3_features(model, processor, image)
    patch_grid = features['patch_grid'][0]  # [H, W, D]
    
    # Normalize
    patch_grid_norm = F.normalize(patch_grid, dim=-1)
    reference_norm = F.normalize(reference_patch, dim=-1)
    
    # Compute similarity map
    similarity = (patch_grid_norm * reference_norm).sum(dim=-1)  # [H, W]
    
    # Find high similarity regions
    detections = []
    mask = similarity > threshold
    
    if mask.any():
        # Connected components to find objects
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask.cpu().numpy())
        
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            coords = np.argwhere(component_mask)
            
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Convert patch coords to image coords
                patch_size = image.size[0] // patch_grid.shape[1]
                box = [
                    x_min * patch_size,
                    y_min * patch_size,
                    (x_max + 1) * patch_size,
                    (y_max + 1) * patch_size
                ]
                
                confidence = similarity[component_mask].mean().item()
                detections.append({'box': box, 'confidence': confidence})
    
    return detections
```

## Integration with Other Models

### SAM 2 Integration
```python
# Assuming SAM 2 is installed
from sam2 import SAM2Predictor

class DINOv3SAM2Pipeline:
    def __init__(self, dinov3_model, dinov3_processor, sam2_checkpoint):
        self.dinov3 = dinov3_model
        self.processor = dinov3_processor
        self.sam2 = SAM2Predictor(sam2_checkpoint)
    
    def segment_by_similarity(self, image, reference_region):
        """Use DINOv3 to find similar regions, then SAM 2 for masks"""
        
        # Find similar regions with DINOv3
        detections = detect_objects_by_similarity(
            self.dinov3, 
            self.processor,
            image,
            reference_region
        )
        
        # Generate masks with SAM 2
        masks = []
        for det in detections:
            box = det['box']
            
            # Set image for SAM 2
            self.sam2.set_image(np.array(image))
            
            # Predict mask from box
            mask, _, _ = self.sam2.predict(
                box=np.array(box),
                multimask_output=False
            )
            
            masks.append({
                'mask': mask,
                'box': box,
                'confidence': det['confidence']
            })
        
        return masks
```

### CLIP Text Integration
```python
import clip

class TextToDINOv3Bridge:
    """Map CLIP text embeddings to DINOv3 feature space"""
    
    def __init__(self, dinov3_model, dinov3_processor):
        self.dinov3 = dinov3_model
        self.processor = dinov3_processor
        
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        
        # Simple projection layer
        self.projection = torch.nn.Linear(
            512,  # CLIP dimension
            dinov3_model.config.hidden_size
        ).to(dinov3_model.device)
    
    def encode_text(self, text):
        """Encode text to DINOv3 feature space"""
        # CLIP encoding
        text_tokens = clip.tokenize([text]).to(self.clip_model.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # Project to DINOv3 space
        dinov3_features = self.projection(text_features)
        dinov3_features = F.normalize(dinov3_features, dim=-1)
        
        return dinov3_features
    
    def find_text_regions(self, image, text, threshold=0.5):
        """Find image regions matching text description"""
        # Get text features in DINOv3 space
        text_features = self.encode_text(text)
        
        # Get image patch features
        image_features = extract_dinov3_features(
            self.dinov3, 
            self.processor, 
            image
        )
        patch_grid = image_features['patch_grid'][0]
        
        # Normalize patches
        patch_grid_norm = F.normalize(patch_grid, dim=-1)
        
        # Compute similarity
        similarity = (patch_grid_norm @ text_features.T).squeeze(-1)
        
        # Return high similarity regions
        return similarity > threshold
```

## Performance Optimization

### Batch Processing
```python
def batch_process_images(model, processor, images, batch_size=8):
    """Efficiently process multiple images"""
    all_features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Process batch
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.cuda.amp.autocast():  # Mixed precision
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state
        
        all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)
```

### Caching Features
```python
import pickle
from pathlib import Path

class DINOv3FeatureCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, image_path):
        """Generate cache filename for image"""
        import hashlib
        hash_val = hashlib.md5(str(image_path).encode()).hexdigest()
        return self.cache_dir / f"{hash_val}.pkl"
    
    def get_features(self, model, processor, image_path):
        """Get features from cache or compute"""
        cache_path = self.get_cache_path(image_path)
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Compute features
        image = Image.open(image_path)
        features = extract_dinov3_features(model, processor, image)
        
        # Cache for future use
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        return features
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM)**
   - Use smaller batch sizes
   - Enable mixed precision (fp16/bf16)
   - Use quantization for large models
   - Use gradient checkpointing for training

2. **Slow Inference**
   - Use torch.compile() for optimization
   - Enable CUDA graphs
   - Use appropriate batch sizes
   - Consider model quantization

3. **Poor Feature Quality**
   - Ensure proper image preprocessing
   - Use appropriate model size for task
   - Check if register tokens are being used
   - Verify normalization is applied correctly

4. **License Issues**
   - Accept DINOv3 license on Hugging Face
   - Use DINOv2 for fully open-source alternative
   - Check commercial usage terms