"""
Image processing utilities for OmniVision
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import cv2


def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Preprocess image for DINOv3 input
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Preprocessed PIL Image
    """
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError("Numpy array must be RGB with shape (H, W, 3)")
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be path, PIL Image, or numpy array")
    
    # Resize if requested
    if target_size is not None:
        if maintain_aspect:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image


def postprocess_features(
    features: torch.Tensor,
    normalize: bool = True,
    to_numpy: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Postprocess extracted features
    
    Args:
        features: Raw features tensor
        normalize: Whether to L2 normalize
        to_numpy: Whether to convert to numpy
        
    Returns:
        Processed features
    """
    if normalize:
        features = torch.nn.functional.normalize(features, dim=-1)
    
    if to_numpy:
        return features.cpu().numpy()
    
    return features


def resize_similarity_map(
    similarity_map: Union[torch.Tensor, np.ndarray],
    target_size: Tuple[int, int],
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize similarity map to target size
    
    Args:
        similarity_map: Input similarity map
        target_size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized similarity map as numpy array
    """
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = similarity_map.cpu().numpy()
    
    # OpenCV resize expects (width, height)
    if interpolation == 'bilinear':
        cv_interp = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        cv_interp = cv2.INTER_NEAREST
    elif interpolation == 'cubic':
        cv_interp = cv2.INTER_CUBIC
    else:
        cv_interp = cv2.INTER_LINEAR
    
    resized = cv2.resize(similarity_map, target_size, interpolation=cv_interp)
    return resized


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = 'jet',
    normalize: bool = True
) -> np.ndarray:
    """
    Apply colormap to heatmap
    
    Args:
        heatmap: Input heatmap
        colormap: Colormap name
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Colored heatmap as RGB array
    """
    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    if colormap == 'jet':
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    elif colormap == 'hot':
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    elif colormap == 'viridis':
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    else:
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored


def overlay_heatmap(
    image: Union[str, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Base image
        heatmap: Heatmap to overlay
        alpha: Blending factor
        colormap: Colormap for heatmap
        
    Returns:
        Overlayed image as RGB array
    """
    # Preprocess image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Resize heatmap to match image
    if heatmap.shape != image_array.shape[:2]:
        heatmap = resize_similarity_map(heatmap, image_array.shape[1::-1])
    
    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(heatmap, colormap)
    
    # Blend
    overlayed = (1 - alpha) * image_array + alpha * colored_heatmap
    
    return overlayed.astype(np.uint8)


def extract_patch_from_coords(
    image: Union[str, Image.Image],
    coords: Tuple[int, int],
    patch_size: int = 64
) -> Image.Image:
    """
    Extract patch from image at given coordinates
    
    Args:
        image: Input image
        coords: (x, y) center coordinates
        patch_size: Size of patch to extract
        
    Returns:
        Extracted patch as PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    x, y = coords
    half_size = patch_size // 2
    
    # Calculate crop box
    left = max(0, x - half_size)
    top = max(0, y - half_size)
    right = min(image.size[0], x + half_size)
    bottom = min(image.size[1], y + half_size)
    
    # Crop patch
    patch = image.crop((left, top, right, bottom))
    
    # Resize to target size if needed
    if patch.size != (patch_size, patch_size):
        patch = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    
    return patch


def draw_bbox(
    image: Union[Image.Image, np.ndarray],
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: (x, y, w, h) bounding box
        color: RGB color
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image.copy()
    
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(
        image_array,
        (x, y),
        (x + w, y + h),
        color[::-1],  # Convert RGB to BGR for OpenCV
        thickness
    )
    
    return image_array


def draw_point(
    image: Union[Image.Image, np.ndarray],
    point: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 5
) -> np.ndarray:
    """
    Draw point on image
    
    Args:
        image: Input image
        point: (x, y) point coordinates
        color: RGB color
        radius: Point radius
        
    Returns:
        Image with point drawn
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image.copy()
    
    x, y = point
    
    # Draw circle
    cv2.circle(
        image_array,
        (x, y),
        radius,
        color[::-1],  # Convert RGB to BGR for OpenCV
        -1  # Filled circle
    )
    
    return image_array