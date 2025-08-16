"""
Segmentation pipeline combining DINOv3 and SAM 2

This module implements the core pipeline that uses DINOv3 for feature extraction
and similarity computation, then SAM 2 for precise segmentation refinement.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from PIL import Image
import cv2
from scipy import ndimage

from ..models.dinov3_backbone import DINOv3Backbone
from ..models.sam2_wrapper import SAM2Wrapper
from ..utils.image import resize_similarity_map, overlay_heatmap


class SegmentationPipeline:
    """
    Pipeline for semantic segmentation using DINOv3 + SAM 2
    
    This class combines DINOv3's self-supervised features with SAM 2's
    segmentation capabilities to provide high-quality semantic segmentation
    from minimal prompts (clicks, patches, or similarity).
    """
    
    def __init__(
        self,
        dinov3_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        sam2_model: str = "tiny",
        device: Optional[str] = None,
        cache_features: bool = True
    ):
        """
        Initialize segmentation pipeline
        
        Args:
            dinov3_model: DINOv3 model identifier
            sam2_model: SAM 2 model size ("tiny", "small", "base", "large")
            device: Device for computation
            cache_features: Whether to cache DINOv3 features
        """
        self.device = device
        self.cache_features = cache_features
        
        # Initialize models
        self.dinov3 = DINOv3Backbone(model_name=dinov3_model, device=device)
        self.sam2 = SAM2Wrapper(model_size=sam2_model, device=device)
        
        self._feature_cache = {} if cache_features else None
        
    def segment_by_click(
        self,
        image: Union[str, Image.Image],
        click_coords: Tuple[int, int],
        similarity_threshold: float = 0.6,
        region_size: int = 1,
        sam2_refine: bool = True
    ) -> Dict:
        """
        Segment objects by clicking on a reference point
        
        Args:
            image: Input image
            click_coords: (x, y) coordinates of click
            similarity_threshold: Threshold for DINOv3 similarity
            region_size: Size of reference region (in patch units)
            sam2_refine: Whether to refine with SAM 2
            
        Returns:
            Dictionary with segmentation results
        """
        # Load image
        if isinstance(image, str):
            img_pil = Image.open(image)
        else:
            img_pil = image
            
        # Extract DINOv3 features
        features = self._get_cached_features(image)
        patch_grid = features['patch_grid'].squeeze(0)  # [H, W, D]
        
        H, W, D = patch_grid.shape
        
        # Convert click coordinates to patch coordinates
        click_x, click_y = click_coords
        patch_x = int(click_x * W / img_pil.size[0])
        patch_y = int(click_y * H / img_pil.size[1])
        
        # Clamp to valid range
        patch_x = max(0, min(patch_x, W - 1))
        patch_y = max(0, min(patch_y, H - 1))
        
        # Extract reference feature
        if region_size == 1:
            reference_feature = patch_grid[patch_y, patch_x, :]
        else:
            half_size = region_size // 2
            y1 = max(0, patch_y - half_size)
            y2 = min(H, patch_y + half_size + 1)
            x1 = max(0, patch_x - half_size)
            x2 = min(W, patch_x + half_size + 1)
            
            patch_region = patch_grid[y1:y2, x1:x2, :]
            reference_feature = patch_region.mean(dim=(0, 1))
        
        # Compute similarity map
        similarity_map = self.dinov3.get_similarity_map(
            image,
            reference_feature,
            normalize=True
        ).cpu().numpy()
        
        result = {
            'click_coords': click_coords,
            'patch_coords': (patch_x, patch_y),
            'similarity_map': similarity_map,
            'reference_feature': reference_feature
        }
        
        if sam2_refine:
            # Use SAM 2 for refinement
            sam2_result = self._refine_with_sam2(
                img_pil, 
                similarity_map, 
                click_coords,
                similarity_threshold
            )
            result.update(sam2_result)
        else:
            # Generate masks from similarity only
            masks = self._masks_from_similarity(
                similarity_map,
                img_pil.size,
                similarity_threshold
            )
            result['masks'] = masks
            
        return result
    
    def segment_by_patch(
        self,
        target_image: Union[str, Image.Image],
        reference_patch: Union[torch.Tensor, Tuple[int, int, int, int]],
        reference_image: Optional[Union[str, Image.Image]] = None,
        similarity_threshold: float = 0.6,
        sam2_refine: bool = True
    ) -> Dict:
        """
        Segment objects using a reference patch
        
        Args:
            target_image: Image to segment
            reference_patch: Either feature tensor or (x, y, w, h) bbox
            reference_image: Source image for bbox (required if reference_patch is bbox)
            similarity_threshold: Similarity threshold
            sam2_refine: Whether to refine with SAM 2
            
        Returns:
            Dictionary with segmentation results
        """
        # Extract reference feature if bbox provided
        if isinstance(reference_patch, (tuple, list)):
            if reference_image is None:
                raise ValueError("reference_image required when reference_patch is bbox")
            
            x, y, w, h = reference_patch
            ref_features = self._get_cached_features(reference_image)
            ref_grid = ref_features['patch_grid'].squeeze(0)  # [H, W, D]
            
            # Convert image coordinates to patch coordinates
            if isinstance(reference_image, str):
                ref_img = Image.open(reference_image)
            else:
                ref_img = reference_image
                
            H, W = ref_grid.shape[:2]
            patch_x1 = int(x * W / ref_img.size[0])
            patch_y1 = int(y * H / ref_img.size[1])
            patch_x2 = int((x + w) * W / ref_img.size[0])
            patch_y2 = int((y + h) * H / ref_img.size[1])
            
            # Extract patch region and average
            patch_region = ref_grid[patch_y1:patch_y2, patch_x1:patch_x2, :]
            reference_feature = patch_region.mean(dim=(0, 1))
        else:
            reference_feature = reference_patch
        
        # Get similarity map
        similarity_map = self.dinov3.get_similarity_map(
            target_image,
            reference_feature,
            normalize=True
        ).cpu().numpy()
        
        # Load target image
        if isinstance(target_image, str):
            target_img = Image.open(target_image)
        else:
            target_img = target_image
        
        result = {
            'similarity_map': similarity_map,
            'reference_feature': reference_feature
        }
        
        if sam2_refine:
            # Use SAM 2 for refinement
            sam2_result = self._refine_with_sam2(
                target_img,
                similarity_map,
                None,  # No specific click point
                similarity_threshold
            )
            result.update(sam2_result)
        else:
            # Generate masks from similarity only
            masks = self._masks_from_similarity(
                similarity_map,
                target_img.size,
                similarity_threshold
            )
            result['masks'] = masks
            
        return result
    
    def segment_everything_similar(
        self,
        image: Union[str, Image.Image],
        reference_features: List[torch.Tensor],
        similarity_threshold: float = 0.5,
        sam2_refine: bool = True
    ) -> Dict:
        """
        Segment all objects similar to reference features
        
        Args:
            image: Input image
            reference_features: List of reference feature vectors
            similarity_threshold: Similarity threshold
            sam2_refine: Whether to refine with SAM 2
            
        Returns:
            Dictionary with all segmentation results
        """
        results = []
        
        for i, ref_feature in enumerate(reference_features):
            result = self.segment_by_patch(
                image,
                ref_feature,
                similarity_threshold=similarity_threshold,
                sam2_refine=sam2_refine
            )
            result['reference_id'] = i
            results.append(result)
        
        return {'segments': results}
    
    def _get_cached_features(self, image: Union[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """Get DINOv3 features with caching"""
        cache_key = str(image) if isinstance(image, str) else id(image)
        
        if self._feature_cache is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        features = self.dinov3.extract_features(image)
        
        if self._feature_cache is not None:
            self._feature_cache[cache_key] = features
            
        return features
    
    def _refine_with_sam2(
        self,
        image: Image.Image,
        similarity_map: np.ndarray,
        click_coords: Optional[Tuple[int, int]],
        threshold: float
    ) -> Dict:
        """Refine similarity-based detection with SAM 2"""
        # Set image in SAM 2
        self.sam2.set_image(image)
        
        # Use SAM 2's built-in similarity refinement
        sam2_result = self.sam2.refine_masks_with_dinov3_similarity(
            similarity_map,
            threshold=threshold,
            min_area=100
        )
        
        # If we have a click point, also try direct SAM 2 prediction
        if click_coords is not None and sam2_result['masks'].size == 0:
            # Fallback to direct point prediction
            point_result = self.sam2.predict_from_points([click_coords])
            if point_result['masks'].size > 0:
                sam2_result = {
                    'masks': point_result['masks'],
                    'scores': point_result['scores'],
                    'boxes': [self._mask_to_bbox(point_result['masks'][0], image.size)]
                }
        
        return {
            'masks': sam2_result['masks'],
            'boxes': sam2_result.get('boxes', []),
            'scores': sam2_result.get('scores', [])
        }
    
    def _masks_from_similarity(
        self,
        similarity_map: np.ndarray,
        image_size: Tuple[int, int],
        threshold: float
    ) -> List[np.ndarray]:
        """Generate masks from similarity map without SAM 2"""
        # Threshold and find connected components
        binary_mask = similarity_map > threshold
        
        if not binary_mask.any():
            return []
        
        # Resize to image size
        h_img, w_img = image_size[1], image_size[0]  # PIL uses (W, H)
        if binary_mask.shape != (h_img, w_img):
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8),
                image_size,
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # Find connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        masks = []
        for label_id in range(1, num_labels + 1):
            component_mask = labeled_mask == label_id
            if component_mask.sum() > 100:  # Minimum area
                masks.append(component_mask)
        
        return masks
    
    def _mask_to_bbox(self, mask: np.ndarray, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert mask to bounding box"""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return (0, 0, 0, 0)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def visualize_segmentation(
        self,
        image: Union[str, Image.Image],
        result: Dict,
        show_similarity: bool = True,
        show_masks: bool = True,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization of segmentation results
        
        Args:
            image: Input image
            result: Segmentation result from pipeline
            show_similarity: Whether to show similarity heatmap
            show_masks: Whether to show segmentation masks
            alpha: Transparency for overlays
            
        Returns:
            Visualization as RGB array
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        img_array = np.array(img)
        
        if show_similarity and 'similarity_map' in result:
            # Overlay similarity heatmap
            similarity_map = result['similarity_map']
            img_array = overlay_heatmap(img_array, similarity_map, alpha=alpha/2)
        
        if show_masks and 'masks' in result:
            masks = result['masks']
            if len(masks) > 0:
                # Overlay masks with different colors
                colors = [
                    (255, 0, 0),    # Red
                    (0, 255, 0),    # Green
                    (0, 0, 255),    # Blue
                    (255, 255, 0),  # Yellow
                    (255, 0, 255),  # Magenta
                ]
                
                for i, mask in enumerate(masks):
                    if isinstance(mask, np.ndarray) and mask.size > 0:
                        color = colors[i % len(colors)]
                        
                        # Create colored mask
                        colored_mask = np.zeros_like(img_array, dtype=np.float64)
                        
                        # Convert boolean mask to proper indexing
                        if mask.ndim == 2:
                            mask_bool = mask.astype(bool)
                        else:
                            mask_bool = mask[0].astype(bool)
                        
                        # Apply color to mask regions
                        colored_mask[mask_bool] = color
                        
                        # Blend with image
                        img_array = img_array.astype(np.float64)
                        img_array = (1 - alpha) * img_array + alpha * colored_mask
        
        return img_array.astype(np.uint8)
    
    def clear_cache(self):
        """Clear feature cache"""
        if self._feature_cache is not None:
            self._feature_cache.clear()
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline"""
        return {
            'dinov3_model': self.dinov3.get_model_info(),
            'sam2_model': self.sam2.get_model_info(),
            'cache_enabled': self.cache_features,
            'cached_items': len(self._feature_cache) if self._feature_cache else 0
        }