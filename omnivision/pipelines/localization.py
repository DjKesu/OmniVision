"""
Localization pipeline for OmniVision

This module implements object localization using DINOv3 features with click-based
interaction and region proposal generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from PIL import Image
import cv2
from scipy import ndimage
from skimage import measure

from ..models.dinov3_backbone import DINOv3Backbone
from .similarity import SimilarityPipeline


class LocalizationPipeline:
    """
    Pipeline for localizing objects using DINOv3 features
    
    Supports click-based localization, region proposals, and heatmap generation.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: Optional[str] = None
    ):
        """
        Initialize localization pipeline
        
        Args:
            model_name: DINOv3 model to use
            device: Device for computation
        """
        self.backbone = DINOv3Backbone(model_name=model_name, device=device)
        self.similarity_pipeline = SimilarityPipeline(model_name=model_name, device=device)
    
    def localize_by_click(
        self,
        image: Union[str, Image.Image],
        click_coords: Tuple[int, int],
        target_image: Optional[Union[str, Image.Image]] = None,
        similarity_threshold: float = 0.6,
        region_size: int = 1
    ) -> Dict:
        """
        Localize objects by clicking on a reference point
        
        Args:
            image: Reference image to click on
            click_coords: (x, y) coordinates of click
            target_image: Target image to search in (if None, use same image)
            similarity_threshold: Threshold for similarity matching
            region_size: Size of region around click (in patch units)
            
        Returns:
            Dictionary with localization results:
            - 'reference_patch': Reference patch coordinates
            - 'similarity_map': Similarity heatmap
            - 'detections': List of detected regions
            - 'reference_feature': Extracted reference feature
        """
        if target_image is None:
            target_image = image
        
        # Load images for size information
        if isinstance(image, str):
            ref_img = Image.open(image)
        else:
            ref_img = image
            
        if isinstance(target_image, str):
            target_img = Image.open(target_image)
        else:
            target_img = target_image
        
        # Extract features from reference image
        ref_features = self.backbone.extract_features(image)
        ref_patch_grid = ref_features['patch_grid'].squeeze(0)  # [H, W, D]
        
        H, W, D = ref_patch_grid.shape
        
        # Convert click coordinates to patch coordinates
        click_x, click_y = click_coords
        patch_x = int(click_x * W / ref_img.size[0])
        patch_y = int(click_y * H / ref_img.size[1])
        
        # Clamp to valid range
        patch_x = max(0, min(patch_x, W - 1))
        patch_y = max(0, min(patch_y, H - 1))
        
        # Extract reference patch feature(s)
        if region_size == 1:
            # Single patch
            reference_feature = ref_patch_grid[patch_y, patch_x, :]
        else:
            # Region around click
            half_size = region_size // 2
            y1 = max(0, patch_y - half_size)
            y2 = min(H, patch_y + half_size + 1)
            x1 = max(0, patch_x - half_size)
            x2 = min(W, patch_x + half_size + 1)
            
            patch_region = ref_patch_grid[y1:y2, x1:x2, :]
            reference_feature = patch_region.mean(dim=(0, 1))
        
        # Get similarity map for target image
        similarity_map = self.backbone.get_similarity_map(
            target_image,
            reference_feature,
            normalize=True
        )
        
        # Find detections
        detections = self._extract_detections_from_heatmap(
            similarity_map,
            target_img.size,
            threshold=similarity_threshold
        )
        
        return {
            'reference_patch': (patch_x, patch_y),
            'reference_coords': click_coords,
            'similarity_map': similarity_map,
            'detections': detections,
            'reference_feature': reference_feature
        }
    
    def localize_by_bbox(
        self,
        image: Union[str, Image.Image],
        bbox: Tuple[int, int, int, int],
        target_image: Optional[Union[str, Image.Image]] = None,
        similarity_threshold: float = 0.6
    ) -> Dict:
        """
        Localize objects using a bounding box as reference
        
        Args:
            image: Reference image
            bbox: (x, y, w, h) bounding box
            target_image: Target image to search in
            similarity_threshold: Threshold for similarity matching
            
        Returns:
            Dictionary with localization results
        """
        if target_image is None:
            target_image = image
            
        # Use similarity pipeline for bbox-based localization
        detections = self.similarity_pipeline.localize_by_patch(
            target_image,
            bbox,
            image,
            threshold=similarity_threshold
        )
        
        # Get similarity map for visualization
        x, y, w, h = bbox
        click_coords = (x + w // 2, y + h // 2)  # Center of bbox
        
        result = self.localize_by_click(
            image,
            click_coords,
            target_image,
            similarity_threshold
        )
        
        # Replace detections with bbox-based ones
        result['detections'] = detections
        result['reference_bbox'] = bbox
        
        return result
    
    def generate_heatmap(
        self,
        image: Union[str, Image.Image],
        reference_feature: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate similarity heatmap
        
        Args:
            image: Target image
            reference_feature: Reference feature vector
            output_size: Output size for heatmap (W, H). If None, use patch grid size
            
        Returns:
            Heatmap as numpy array
        """
        similarity_map = self.backbone.get_similarity_map(
            image,
            reference_feature,
            normalize=True
        )
        
        heatmap = similarity_map.cpu().numpy()
        
        # Resize if requested
        if output_size is not None:
            heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return heatmap
    
    def _extract_detections_from_heatmap(
        self,
        similarity_map: torch.Tensor,
        image_size: Tuple[int, int],
        threshold: float = 0.6,
        min_area: int = 4,
        max_detections: int = 20
    ) -> List[Dict]:
        """
        Extract object detections from similarity heatmap
        
        Args:
            similarity_map: Similarity heatmap [H, W]
            image_size: (width, height) of original image
            threshold: Similarity threshold
            min_area: Minimum area in patch units
            max_detections: Maximum number of detections
            
        Returns:
            List of detection dictionaries
        """
        heatmap = similarity_map.cpu().numpy()
        H, W = heatmap.shape
        img_w, img_h = image_size
        
        # Threshold the heatmap
        binary_mask = heatmap > threshold
        
        if not binary_mask.any():
            return []
        
        # Find connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        detections = []
        
        for label_id in range(1, num_labels + 1):
            # Get component mask
            component_mask = labeled_mask == label_id
            
            # Check minimum area
            if component_mask.sum() < min_area:
                continue
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Convert patch coordinates to image coordinates
            bbox_x1 = int(x_min * img_w / W)
            bbox_y1 = int(y_min * img_h / H)
            bbox_x2 = int((x_max + 1) * img_w / W)
            bbox_y2 = int((y_max + 1) * img_h / H)
            
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1
            
            # Calculate average similarity score
            avg_similarity = heatmap[component_mask].mean()
            max_similarity = heatmap[component_mask].max()
            
            # Calculate centroid
            centroid_y, centroid_x = ndimage.center_of_mass(component_mask)
            center_x = int(centroid_x * img_w / W)
            center_y = int(centroid_y * img_h / H)
            
            detections.append({
                'bbox': (bbox_x1, bbox_y1, bbox_w, bbox_h),
                'center': (center_x, center_y),
                'similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'area': int(component_mask.sum()),
                'patch_bbox': (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            })
        
        # Sort by similarity and return top detections
        detections.sort(key=lambda x: x['max_similarity'], reverse=True)
        return detections[:max_detections]
    
    def multi_scale_localization(
        self,
        image: Union[str, Image.Image],
        click_coords: Tuple[int, int],
        target_image: Optional[Union[str, Image.Image]] = None,
        scales: List[int] = [1, 2, 3],
        similarity_threshold: float = 0.6
    ) -> Dict:
        """
        Perform localization at multiple scales
        
        Args:
            image: Reference image
            click_coords: Click coordinates
            target_image: Target image
            scales: List of region sizes to try
            similarity_threshold: Similarity threshold
            
        Returns:
            Dictionary with multi-scale results
        """
        if target_image is None:
            target_image = image
        
        results = {}
        all_detections = []
        
        for scale in scales:
            result = self.localize_by_click(
                image,
                click_coords,
                target_image,
                similarity_threshold,
                region_size=scale
            )
            
            results[f'scale_{scale}'] = result
            
            # Add scale information to detections
            for det in result['detections']:
                det['scale'] = scale
                all_detections.append(det)
        
        # Combine and deduplicate detections
        combined_detections = self._combine_multi_scale_detections(all_detections)
        
        return {
            'scale_results': results,
            'combined_detections': combined_detections,
            'click_coords': click_coords
        }
    
    def _combine_multi_scale_detections(
        self,
        detections: List[Dict],
        overlap_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Combine detections from multiple scales using NMS
        
        Args:
            detections: List of all detections
            overlap_threshold: IoU threshold for combining
            
        Returns:
            List of combined detections
        """
        if not detections:
            return []
        
        # Sort by similarity
        detections.sort(key=lambda x: x['similarity'], reverse=True)
        
        combined = []
        
        for det in detections:
            # Check if this detection overlaps significantly with existing ones
            overlap = False
            
            for existing in combined:
                iou = self._compute_bbox_iou(det['bbox'], existing['bbox'])
                if iou > overlap_threshold:
                    # Keep the one with higher similarity
                    if det['similarity'] > existing['similarity']:
                        combined.remove(existing)
                        combined.append(det)
                    overlap = True
                    break
            
            if not overlap:
                combined.append(det)
        
        return sorted(combined, key=lambda x: x['similarity'], reverse=True)
    
    def _compute_bbox_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0