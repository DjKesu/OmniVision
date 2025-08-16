"""
Similarity computation pipeline for OmniVision

This module implements various similarity-based operations using DINOv3 features,
including correspondence matching, object localization, and visual search.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from PIL import Image
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..models.dinov3_backbone import DINOv3Backbone


class SimilarityPipeline:
    """
    Pipeline for computing visual similarities using DINOv3 features
    
    This class provides high-level methods for finding correspondences,
    localizing objects by example, and performing visual search.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: Optional[str] = None,
        cache_features: bool = True
    ):
        """
        Initialize similarity pipeline
        
        Args:
            model_name: DINOv3 model to use
            device: Device for computation
            cache_features: Whether to cache extracted features
        """
        self.backbone = DINOv3Backbone(model_name=model_name, device=device)
        self.cache_features = cache_features
        self._feature_cache = {}
        
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image features"""
        return f"{image_path}_{self.backbone.model_name}"
    
    def _extract_cached_features(self, image: Union[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """Extract features with caching support"""
        # Only cache for file paths
        if isinstance(image, str) and self.cache_features:
            cache_key = self._get_cache_key(image)
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
        
        # Extract features
        features = self.backbone.extract_features(image)
        
        # Cache if applicable
        if isinstance(image, str) and self.cache_features:
            self._feature_cache[cache_key] = features
            
        return features
    
    def compute_patch_similarity(
        self,
        image1: Union[str, Image.Image],
        image2: Union[str, Image.Image],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise similarity between all patches in two images
        
        Args:
            image1: First image
            image2: Second image  
            normalize: Whether to normalize features
            
        Returns:
            Similarity matrix [H1*W1, H2*W2]
        """
        # Extract features
        features1 = self._extract_cached_features(image1)
        features2 = self._extract_cached_features(image2)
        
        # Get patch features
        patches1 = features1['patches'].squeeze(0)  # [N1, D]
        patches2 = features2['patches'].squeeze(0)  # [N2, D]
        
        # Normalize if requested
        if normalize:
            patches1 = F.normalize(patches1, dim=-1)
            patches2 = F.normalize(patches2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.mm(patches1, patches2.T)  # [N1, N2]
        
        return similarity.cpu().numpy()
    
    def find_correspondences(
        self,
        image1: Union[str, Image.Image], 
        image2: Union[str, Image.Image],
        num_matches: int = 10,
        min_similarity: float = 0.5,
        use_mutual_nearest: bool = True
    ) -> List[Dict]:
        """
        Find corresponding patches between two images
        
        Args:
            image1: First image
            image2: Second image
            num_matches: Maximum number of correspondences to return
            min_similarity: Minimum similarity threshold
            use_mutual_nearest: Whether to use mutual nearest neighbor filtering
            
        Returns:
            List of correspondence dictionaries with keys:
            - 'coord1': (x, y) in image1
            - 'coord2': (x, y) in image2  
            - 'similarity': similarity score
        """
        # Get image sizes for coordinate conversion
        if isinstance(image1, str):
            img1_pil = Image.open(image1)
        else:
            img1_pil = image1
            
        if isinstance(image2, str):
            img2_pil = Image.open(image2)
        else:
            img2_pil = image2
        
        # Extract features
        features1 = self._extract_cached_features(image1)
        features2 = self._extract_cached_features(image2)
        
        # Get patch grids
        grid1 = features1['patch_grid'].squeeze(0)  # [H1, W1, D]
        grid2 = features2['patch_grid'].squeeze(0)  # [H2, W2, D]
        
        H1, W1, D = grid1.shape
        H2, W2, D = grid2.shape
        
        # Flatten and normalize
        patches1 = F.normalize(grid1.reshape(-1, D), dim=-1)  # [H1*W1, D]
        patches2 = F.normalize(grid2.reshape(-1, D), dim=-1)  # [H2*W2, D]
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(patches1, patches2.T)  # [H1*W1, H2*W2]
        
        # Find correspondences
        correspondences = []
        
        if use_mutual_nearest:
            # Mutual nearest neighbors
            nn1_to_2 = torch.argmax(similarity_matrix, dim=1)  # [H1*W1]
            nn2_to_1 = torch.argmax(similarity_matrix, dim=0)  # [H2*W2]
            
            # Check for mutual nearest neighbors
            for i1 in range(len(nn1_to_2)):
                i2 = nn1_to_2[i1].item()
                if nn2_to_1[i2].item() == i1:  # Mutual NN
                    sim = similarity_matrix[i1, i2].item()
                    if sim >= min_similarity:
                        # Convert to 2D coordinates
                        y1, x1 = divmod(i1, W1)
                        y2, x2 = divmod(i2, W2)
                        
                        # Convert to image coordinates
                        coord1 = (
                            int(x1 * img1_pil.size[0] / W1),
                            int(y1 * img1_pil.size[1] / H1)
                        )
                        coord2 = (
                            int(x2 * img2_pil.size[0] / W2),
                            int(y2 * img2_pil.size[1] / H2)
                        )
                        
                        correspondences.append({
                            'coord1': coord1,
                            'coord2': coord2,
                            'similarity': sim,
                            'patch_coord1': (x1, y1),
                            'patch_coord2': (x2, y2)
                        })
        else:
            # Top-k correspondences
            similarities_flat = similarity_matrix.flatten()
            top_indices = torch.topk(similarities_flat, min(num_matches * 5, len(similarities_flat))).indices
            
            for idx in top_indices:
                i1 = idx // similarity_matrix.shape[1]
                i2 = idx % similarity_matrix.shape[1]
                sim = similarity_matrix[i1, i2].item()
                
                if sim >= min_similarity:
                    # Convert to 2D coordinates  
                    y1, x1 = divmod(i1.item(), W1)
                    y2, x2 = divmod(i2.item(), W2)
                    
                    # Convert to image coordinates
                    coord1 = (
                        int(x1 * img1_pil.size[0] / W1),
                        int(y1 * img1_pil.size[1] / H1)
                    )
                    coord2 = (
                        int(x2 * img2_pil.size[0] / W2),
                        int(y2 * img2_pil.size[1] / H2)
                    )
                    
                    correspondences.append({
                        'coord1': coord1,
                        'coord2': coord2,
                        'similarity': sim,
                        'patch_coord1': (x1, y1),
                        'patch_coord2': (x2, y2)
                    })
        
        # Sort by similarity and return top matches
        correspondences.sort(key=lambda x: x['similarity'], reverse=True)
        return correspondences[:num_matches]
    
    def localize_by_patch(
        self,
        target_image: Union[str, Image.Image],
        reference_patch: Union[torch.Tensor, Tuple[int, int, int, int]],
        reference_image: Optional[Union[str, Image.Image]] = None,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Localize regions in target image similar to a reference patch
        
        Args:
            target_image: Image to search in
            reference_patch: Either a feature tensor or (x, y, w, h) bbox
            reference_image: Source image for bbox (required if reference_patch is bbox)
            threshold: Similarity threshold
            top_k: Maximum number of detections
            
        Returns:
            List of detection dictionaries with keys:
            - 'bbox': (x, y, w, h) bounding box
            - 'similarity': similarity score
            - 'center': (x, y) center coordinates
        """
        # Extract reference feature if bbox provided
        if isinstance(reference_patch, (tuple, list)):
            if reference_image is None:
                raise ValueError("reference_image required when reference_patch is bbox")
            
            x, y, w, h = reference_patch
            ref_features = self._extract_cached_features(reference_image)
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
            reference_feature = patch_region.mean(dim=(0, 1))  # [D]
        else:
            reference_feature = reference_patch
        
        # Get similarity map
        similarity_map = self.backbone.get_similarity_map(
            target_image, 
            reference_feature,
            normalize=True
        )  # [H, W]
        
        # Find peaks above threshold
        similarity_np = similarity_map.cpu().numpy()
        peaks = self._find_similarity_peaks(similarity_np, threshold, top_k)
        
        # Convert to detections
        if isinstance(target_image, str):
            target_img = Image.open(target_image)
        else:
            target_img = target_image
            
        detections = []
        H, W = similarity_map.shape
        
        for peak_y, peak_x, score in peaks:
            # Convert patch coordinates to image coordinates
            center_x = int(peak_x * target_img.size[0] / W)
            center_y = int(peak_y * target_img.size[1] / H)
            
            # Estimate bounding box size (simple heuristic)
            patch_size = self.backbone.patch_size
            bbox_w = patch_size * 2
            bbox_h = patch_size * 2
            
            bbox = (
                max(0, center_x - bbox_w // 2),
                max(0, center_y - bbox_h // 2),
                bbox_w,
                bbox_h
            )
            
            detections.append({
                'bbox': bbox,
                'center': (center_x, center_y),
                'similarity': score,
                'patch_coord': (peak_x, peak_y)
            })
        
        return detections
    
    def _find_similarity_peaks(
        self,
        similarity_map: np.ndarray,
        threshold: float,
        max_peaks: int
    ) -> List[Tuple[int, int, float]]:
        """Find peaks in similarity map using non-maximum suppression"""
        # Apply threshold
        mask = similarity_map > threshold
        
        if not mask.any():
            return []
        
        # Find local maxima
        peaks = []
        kernel_size = 3
        
        for y in range(kernel_size//2, similarity_map.shape[0] - kernel_size//2):
            for x in range(kernel_size//2, similarity_map.shape[1] - kernel_size//2):
                if mask[y, x]:
                    # Check if it's a local maximum
                    local_region = similarity_map[
                        y-kernel_size//2:y+kernel_size//2+1,
                        x-kernel_size//2:x+kernel_size//2+1
                    ]
                    
                    if similarity_map[y, x] == local_region.max():
                        peaks.append((y, x, similarity_map[y, x]))
        
        # Sort by score and return top peaks
        peaks.sort(key=lambda x: x[2], reverse=True)
        return peaks[:max_peaks]
    
    def compute_image_similarity(
        self,
        image1: Union[str, Image.Image],
        image2: Union[str, Image.Image],
        method: str = "cls"
    ) -> float:
        """
        Compute overall similarity between two images
        
        Args:
            image1: First image
            image2: Second image
            method: Similarity method ("cls", "patches", "mixed")
            
        Returns:
            Similarity score (0-1)
        """
        features1 = self._extract_cached_features(image1)
        features2 = self._extract_cached_features(image2)
        
        if method == "cls":
            # Use CLS tokens
            feat1 = F.normalize(features1['cls'], dim=-1)
            feat2 = F.normalize(features2['cls'], dim=-1)
            similarity = torch.sum(feat1 * feat2, dim=-1).item()
            
        elif method == "patches":
            # Use average patch features
            patches1 = features1['patches'].mean(dim=1)  # [1, D]
            patches2 = features2['patches'].mean(dim=1)  # [1, D]
            
            feat1 = F.normalize(patches1, dim=-1)
            feat2 = F.normalize(patches2, dim=-1)
            similarity = torch.sum(feat1 * feat2, dim=-1).item()
            
        elif method == "mixed":
            # Combine CLS and patch features
            cls_sim = self.compute_image_similarity(image1, image2, "cls")
            patch_sim = self.compute_image_similarity(image1, image2, "patches")
            similarity = 0.6 * cls_sim + 0.4 * patch_sim
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
    
    def clear_cache(self):
        """Clear feature cache"""
        self._feature_cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about feature cache"""
        return {
            'num_cached': len(self._feature_cache),
            'cache_enabled': self.cache_features
        }