"""
Text-guided segmentation pipeline using CLIP + DINOv3 + SAM 2

This module implements text-to-vision bridging for semantic segmentation,
allowing users to segment objects using natural language descriptions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from PIL import Image
import cv2
import logging

from ..models.dinov3_backbone import DINOv3Backbone
from ..models.sam2_wrapper import SAM2Wrapper
from ..models.clip_wrapper import CLIPWrapper
from ..models.simple_fusion import SimpleCrossModalFusion
from ..models.improved_fusion import CLIPAlignedFusion
from ..models.trident_fusion import TridentInspiredFusion
from ..utils.image import resize_similarity_map

logger = logging.getLogger("omnivision")


class TextGuidedPipeline:
    """
    Pipeline for text-guided object segmentation
    
    Combines CLIP for text understanding, DINOv3 for visual features,
    and SAM 2 for precise segmentation to enable natural language
    driven object detection and segmentation.
    """
    
    def __init__(
        self,
        dinov3_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        clip_model: str = "ViT-B/32",
        sam2_model: str = "tiny",
        device: Optional[str] = None,
        use_improved_fusion: bool = True
    ):
        """
        Initialize text-guided pipeline
        
        Args:
            dinov3_model: DINOv3 model identifier
            clip_model: CLIP model name
            sam2_model: SAM 2 model size
            device: Device for computation
            use_improved_fusion: Whether to use improved cross-modal fusion
        """
        self.device = device
        self.use_improved_fusion = use_improved_fusion
        self.text_similarity_weight = 0.6  # Weight for CLIP vs DINOv3 similarity
        
        # Initialize models
        self.dinov3 = DINOv3Backbone(model_name=dinov3_model, device=device)
        self.clip = CLIPWrapper(model_name=clip_model, device=device)
        self.sam2 = SAM2Wrapper(model_size=sam2_model, device=device)
        
        # Initialize fusion system
        if use_improved_fusion:
            # Use Trident-inspired fusion for much better alignment
            self.fusion_system = TridentInspiredFusion(
                clip_model=self.clip.model,
                sam_model=self.sam2,
                device=device
            )
        else:
            self.fusion_system = None
        
        logger.info(f"Text-guided pipeline initialized")
        logger.info(f"DINOv3: {dinov3_model}, CLIP: {clip_model}, SAM2: {sam2_model}")
        logger.info(f"Improved fusion: {use_improved_fusion}")
    
    def segment_by_text(
        self,
        image: Union[str, Image.Image],
        text_query: str,
        similarity_threshold: float = 0.3,
        sam2_refine: bool = True,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Segment objects in image using text description
        
        Args:
            image: Input image
            text_query: Text description of object to segment
            similarity_threshold: Threshold for similarity-based segmentation
            sam2_refine: Whether to refine with SAM 2
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Segmentation results dictionary
        """
        logger.info(f"Text-guided segmentation: '{text_query}'")
        
        # Load image
        if isinstance(image, str):
            img_pil = Image.open(image)
        else:
            img_pil = image
        
        if self.use_improved_fusion and self.fusion_system is not None:
            # Use improved cross-modal fusion
            combined_similarity = self._get_improved_similarity(img_pil, text_query)
            # For intermediate results, also compute basic similarities
            if return_intermediate:
                clip_similarity = self._get_clip_similarity_map(img_pil, text_query)
                dinov3_similarity = self._get_dinov3_text_similarity(img_pil, text_query)
        else:
            # Fallback to basic fusion
            clip_similarity = self._get_clip_similarity_map(img_pil, text_query)
            dinov3_similarity = self._get_dinov3_text_similarity(img_pil, text_query)
            combined_similarity = self._combine_similarities(clip_similarity, dinov3_similarity)
        
        result = {
            'text_query': text_query,
            'combined_similarity': combined_similarity
        }
        
        if return_intermediate:
            result.update({
                'clip_similarity': clip_similarity,
                'dinov3_similarity': dinov3_similarity
            })
        
        if sam2_refine:
            # Use SAM 2 for refinement
            sam2_result = self._refine_with_sam2(
                img_pil,
                combined_similarity,
                similarity_threshold
            )
            result.update(sam2_result)
        else:
            # Generate masks from similarity only
            masks = self._masks_from_similarity(
                combined_similarity,
                img_pil.size,
                similarity_threshold
            )
            result['masks'] = masks
        
        return result
    
    def search_and_segment(
        self,
        image: Union[str, Image.Image],
        text_queries: List[str],
        top_k: int = 3,
        segment_top: bool = True
    ) -> Dict:
        """
        Search for multiple objects and segment the most relevant ones
        
        Args:
            image: Input image
            text_queries: List of text descriptions to search for
            top_k: Number of top queries to segment
            segment_top: Whether to segment the top results
            
        Returns:
            Search and segmentation results
        """
        logger.info(f"Searching for {len(text_queries)} queries in image")
        
        # Load image
        if isinstance(image, str):
            img_pil = Image.open(image)
        else:
            img_pil = image
        
        # Compute similarities for all queries
        query_results = []
        for query in text_queries:
            # Get overall image-text similarity
            overall_sim = self.clip.compute_text_image_similarity(query, img_pil)
            overall_score = float(overall_sim.squeeze())
            
            query_results.append({
                'query': query,
                'overall_score': overall_score
            })
        
        # Sort by overall similarity
        query_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        result = {
            'query_rankings': query_results,
            'segmentations': {}
        }
        
        # Segment top-k queries if requested
        if segment_top:
            for i, query_info in enumerate(query_results[:top_k]):
                query = query_info['query']
                logger.info(f"Segmenting top-{i+1} query: '{query}'")
                
                seg_result = self.segment_by_text(
                    img_pil,
                    query,
                    sam2_refine=True,
                    return_intermediate=False
                )
                
                result['segmentations'][query] = seg_result
        
        return result
    
    def interactive_refinement(
        self,
        image: Union[str, Image.Image],
        initial_text: str,
        refinement_texts: List[str],
        combination_mode: str = "union"
    ) -> Dict:
        """
        Refine segmentation using multiple text descriptions
        
        Args:
            image: Input image
            initial_text: Initial text query
            refinement_texts: Additional text descriptions for refinement
            combination_mode: How to combine multiple queries ("union", "intersection", "weighted")
            
        Returns:
            Refined segmentation results
        """
        logger.info(f"Interactive refinement starting with: '{initial_text}'")
        
        # Load image
        if isinstance(image, str):
            img_pil = Image.open(image)
        else:
            img_pil = image
        
        # Get initial segmentation
        initial_result = self.segment_by_text(
            img_pil,
            initial_text,
            return_intermediate=True
        )
        
        # Process refinement texts
        refinement_results = []
        for ref_text in refinement_texts:
            ref_result = self.segment_by_text(
                img_pil,
                ref_text,
                return_intermediate=True
            )
            refinement_results.append(ref_result)
        
        # Combine results based on mode
        if combination_mode == "union":
            combined_similarity = initial_result['combined_similarity'].copy()
            for ref_result in refinement_results:
                combined_similarity = np.maximum(
                    combined_similarity,
                    ref_result['combined_similarity']
                )
        elif combination_mode == "intersection":
            combined_similarity = initial_result['combined_similarity'].copy()
            for ref_result in refinement_results:
                combined_similarity = np.minimum(
                    combined_similarity,
                    ref_result['combined_similarity']
                )
        elif combination_mode == "weighted":
            # Weight by query importance (could be learned)
            weights = [1.0] + [0.7] * len(refinement_results)
            total_weight = sum(weights)
            
            combined_similarity = weights[0] * initial_result['combined_similarity']
            for i, ref_result in enumerate(refinement_results):
                combined_similarity += weights[i+1] * ref_result['combined_similarity']
            combined_similarity /= total_weight
        
        # Final segmentation with SAM 2
        sam2_result = self._refine_with_sam2(img_pil, combined_similarity, 0.3)
        
        return {
            'initial_text': initial_text,
            'refinement_texts': refinement_texts,
            'combination_mode': combination_mode,
            'initial_result': initial_result,
            'refinement_results': refinement_results,
            'final_similarity': combined_similarity,
            'final_masks': sam2_result['masks'],
            'final_boxes': sam2_result.get('boxes', [])
        }
    
    def _get_improved_similarity(self, image: Image.Image, text: str) -> np.ndarray:
        """
        Get improved cross-modal similarity using Trident-inspired fusion
        """
        # Extract DINOv3 features
        dinov3_features = self.dinov3.extract_features(image)
        patch_features = dinov3_features['patches']  # [1, N, D]
        
        # Extract CLIP text features
        clip_text_features = self.clip.encode_text(text)  # [1, D_clip]
        
        # Use Trident-inspired fusion system (pass image for SAM correlation)
        aligned_similarity = self.fusion_system.compute_aligned_similarity(
            patch_features,
            clip_text_features,
            image,  # Pass PIL image for SAM
            image.size
        )
        
        logger.info(f"Trident-inspired similarity range: [{aligned_similarity.min():.3f}, {aligned_similarity.max():.3f}]")
        
        return aligned_similarity
    
    def _get_clip_similarity_map(self, image: Image.Image, text: str) -> np.ndarray:
        """Get CLIP-based similarity map (coarse level)"""
        # For ViT models, we can get patch-level features, but this requires model surgery
        # For now, return a uniform similarity based on overall image-text similarity
        
        overall_sim = self.clip.compute_text_image_similarity(text, image)
        overall_score = float(overall_sim.squeeze())
        
        # Create a uniform similarity map
        # In a full implementation, this would use CLIP's spatial attention
        h, w = 14, 14  # Standard patch grid size
        similarity_map = np.full((h, w), overall_score)
        
        logger.info(f"CLIP overall similarity: {overall_score:.3f}")
        return similarity_map
    
    def _get_dinov3_text_similarity(self, image: Image.Image, text: str) -> np.ndarray:
        """Get DINOv3 features and compute text-guided similarity"""
        # Extract DINOv3 features
        features = self.dinov3.extract_features(image)
        patch_grid = features['patch_grid'].squeeze(0)  # [H, W, D]
        H, W, D = patch_grid.shape
        
        # Get CLIP text features
        text_features = self.clip.encode_text(text)  # [1, D_clip]
        
        # For cross-modal alignment, we need to project DINOv3 features to CLIP space
        # This is a simplified approach - in practice, you'd train alignment layers
        
        # Flatten DINOv3 features
        patches_flat = patch_grid.view(-1, D)  # [H*W, D_dinov3]
        
        # Ensure consistent dtypes
        patches_flat = patches_flat.float()
        text_features = text_features.float()
        
        # Simple projection using L2 normalization (placeholder)
        patches_norm = F.normalize(patches_flat, dim=1)
        text_norm = F.normalize(text_features, dim=1)
        
        # Compute similarity using a simple approach
        # In practice, you'd need proper cross-modal alignment
        if D == text_features.shape[1]:
            # Dimensions match, can compute directly
            similarity = torch.mm(patches_norm, text_norm.T).squeeze(1)
        else:
            # Dimensions don't match, use a simple projection
            # Project to smaller dimension
            target_dim = min(D, text_features.shape[1])
            patch_proj = patches_norm[:, :target_dim]
            text_proj = text_norm[:, :target_dim]
            similarity = torch.mm(patch_proj, text_proj.T).squeeze(1)
        
        # Reshape back to spatial grid
        similarity_map = similarity.view(H, W).cpu().numpy()
        
        logger.info(f"DINOv3-text similarity range: [{similarity_map.min():.3f}, {similarity_map.max():.3f}]")
        return similarity_map
    
    def _combine_similarities(self, clip_sim: np.ndarray, dinov3_sim: np.ndarray) -> np.ndarray:
        """Combine CLIP and DINOv3 similarities"""
        # Ensure same shape
        if clip_sim.shape != dinov3_sim.shape:
            # Resize to match
            target_shape = dinov3_sim.shape
            clip_sim = cv2.resize(clip_sim, (target_shape[1], target_shape[0]))
        
        # Normalize both to [0, 1]
        clip_norm = (clip_sim - clip_sim.min()) / (clip_sim.max() - clip_sim.min() + 1e-8)
        dinov3_norm = (dinov3_sim - dinov3_sim.min()) / (dinov3_sim.max() - dinov3_sim.min() + 1e-8)
        
        # Weighted combination
        combined = (self.text_similarity_weight * clip_norm + 
                   (1 - self.text_similarity_weight) * dinov3_norm)
        
        logger.info(f"Combined similarity range: [{combined.min():.3f}, {combined.max():.3f}]")
        return combined
    
    def _refine_with_sam2(self, image: Image.Image, similarity_map: np.ndarray, threshold: float) -> Dict:
        """Refine similarity-based detection with SAM 2"""
        # Set image in SAM 2
        self.sam2.set_image(image)
        
        # Use SAM 2's similarity refinement
        sam2_result = self.sam2.refine_masks_with_dinov3_similarity(
            similarity_map,
            threshold=threshold,
            min_area=100
        )
        
        return {
            'masks': sam2_result['masks'],
            'boxes': sam2_result.get('boxes', []),
            'scores': sam2_result.get('scores', [])
        }
    
    def _masks_from_similarity(self, similarity_map: np.ndarray, image_size: Tuple[int, int], threshold: float) -> List[np.ndarray]:
        """Generate masks from similarity map without SAM 2"""
        from scipy import ndimage
        
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
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline"""
        return {
            'dinov3_model': self.dinov3.get_model_info(),
            'clip_model': self.clip.get_model_info(),
            'sam2_model': self.sam2.get_model_info(),
            'text_similarity_weight': self.text_similarity_weight
        }