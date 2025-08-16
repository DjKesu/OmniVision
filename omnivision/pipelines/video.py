"""
Video processing pipeline combining DINOv3 and SAM 2

This module implements temporal tracking and segmentation for video sequences,
using DINOv3 for cross-frame feature consistency and SAM 2 for frame-wise segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from PIL import Image
import cv2
from pathlib import Path
import logging

from ..models.dinov3_backbone import DINOv3Backbone
from ..models.sam2_wrapper import SAM2Wrapper
from ..utils.image import resize_similarity_map

logger = logging.getLogger("omnivision")


class VideoPipeline:
    """
    Pipeline for video object tracking and segmentation using DINOv3 + SAM 2
    
    This class provides temporal consistency by leveraging DINOv3's semantic features
    across frames and SAM 2's robust video tracking capabilities.
    """
    
    def __init__(
        self,
        dinov3_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        sam2_model: str = "tiny",
        device: Optional[str] = None,
        temporal_consistency_weight: float = 0.3
    ):
        """
        Initialize video processing pipeline
        
        Args:
            dinov3_model: DINOv3 model identifier
            sam2_model: SAM 2 model size
            device: Device for computation
            temporal_consistency_weight: Weight for temporal consistency in tracking
        """
        self.device = device
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # Initialize models
        self.dinov3 = DINOv3Backbone(model_name=dinov3_model, device=device)
        self.sam2 = SAM2Wrapper(model_size=sam2_model, device=device)
        
        logger.info(f"Video pipeline initialized with DINOv3: {dinov3_model}, SAM2: {sam2_model}")
    
    def load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load video frames from file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            
        Returns:
            List of video frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frames
    
    def extract_video_features(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Extract DINOv3 features for all video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of feature dictionaries for each frame
        """
        frame_features = []
        
        logger.info(f"Extracting features for {len(frames)} frames...")
        for i, frame in enumerate(frames):
            # Convert numpy array to PIL Image
            frame_pil = Image.fromarray(frame)
            
            # Extract features
            features = self.dinov3.extract_features(frame_pil)
            frame_features.append(features)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(frames)} frames")
        
        return frame_features
    
    def track_by_similarity(
        self,
        frames: List[np.ndarray],
        reference_frame_idx: int,
        reference_coords: Tuple[int, int],
        similarity_threshold: float = 0.6,
        temporal_smoothing: bool = True
    ) -> Dict:
        """
        Track object across video using DINOv3 similarity
        
        Args:
            frames: Video frames
            reference_frame_idx: Index of reference frame
            reference_coords: (x, y) coordinates in reference frame
            similarity_threshold: Similarity threshold for tracking
            temporal_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Tracking results with similarity maps and detected regions
        """
        # Extract features for all frames
        frame_features = self.extract_video_features(frames)
        
        # Get reference feature
        ref_features = frame_features[reference_frame_idx]
        ref_patch_grid = ref_features['patch_grid'].squeeze(0)  # [H, W, D]
        H, W, D = ref_patch_grid.shape
        
        # Convert reference coordinates to patch coordinates
        ref_frame = frames[reference_frame_idx]
        ref_x, ref_y = reference_coords
        patch_x = int(ref_x * W / ref_frame.shape[1])
        patch_y = int(ref_y * H / ref_frame.shape[0])
        
        # Clamp to valid range
        patch_x = max(0, min(patch_x, W - 1))
        patch_y = max(0, min(patch_y, H - 1))
        
        # Extract reference feature vector
        reference_feature = ref_patch_grid[patch_y, patch_x, :]
        
        # Track across all frames
        tracking_results = {
            'reference_frame': reference_frame_idx,
            'reference_coords': reference_coords,
            'similarity_maps': [],
            'detections': [],
            'temporal_consistency': []
        }
        
        prev_detection = None
        
        for frame_idx, features in enumerate(frame_features):
            # Compute similarity map
            patch_grid = features['patch_grid'].squeeze(0)  # [H, W, D]
            
            # Flatten for similarity computation
            patches_flat = patch_grid.view(-1, D)  # [H*W, D]
            reference_flat = reference_feature.unsqueeze(0)  # [1, D]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(patches_flat, reference_flat, dim=1)
            similarity_map = similarity.view(H, W).cpu().numpy()
            
            # Apply temporal smoothing if enabled
            if temporal_smoothing and prev_detection is not None and frame_idx > 0:
                # Weight current similarity with previous detection
                prev_sim_map = tracking_results['similarity_maps'][frame_idx - 1]
                similarity_map = (1 - self.temporal_consistency_weight) * similarity_map + \
                               self.temporal_consistency_weight * prev_sim_map
            
            tracking_results['similarity_maps'].append(similarity_map)
            
            # Find detections above threshold
            detections = self._find_detections_in_similarity_map(
                similarity_map,
                frames[frame_idx].shape[:2],
                similarity_threshold
            )
            
            tracking_results['detections'].append(detections)
            
            # Compute temporal consistency score
            if frame_idx > 0:
                consistency_score = self._compute_temporal_consistency(
                    tracking_results['detections'][frame_idx - 1],
                    detections
                )
                tracking_results['temporal_consistency'].append(consistency_score)
            
            if detections:
                prev_detection = detections[0]  # Use strongest detection
        
        return tracking_results
    
    def track_with_sam2(
        self,
        frames: List[np.ndarray],
        initial_prompts: Dict[int, Dict],
        use_dinov3_guidance: bool = True
    ) -> Dict:
        """
        Track objects using SAM 2 with optional DINOv3 guidance
        
        Args:
            frames: Video frames
            initial_prompts: Initial prompts for SAM 2
            use_dinov3_guidance: Whether to use DINOv3 for guidance
            
        Returns:
            Complete tracking results
        """
        # Initialize SAM 2 video predictor
        self.sam2.init_video_predictor()
        
        if self.sam2.video_predictor is None:
            logger.warning("SAM 2 video predictor not available, falling back to frame-by-frame processing")
            segments = self._track_frame_by_frame(frames, initial_prompts)
            return {
                'segments': segments,
                'inference_state': None,
                'num_frames': len(frames),
                'initial_prompts': initial_prompts
            }
        
        # Start video inference
        inference_state = self.sam2.start_video_inference(frames)
        
        # Add initial prompts
        for frame_idx, prompt_data in initial_prompts.items():
            object_id = prompt_data.get('object_id', 1)
            
            if 'points' in prompt_data:
                self.sam2.add_video_points(
                    inference_state,
                    frame_idx,
                    prompt_data['points'],
                    prompt_data.get('labels'),
                    object_id
                )
            
            if 'boxes' in prompt_data:
                for box in prompt_data['boxes']:
                    self.sam2.add_video_box(
                        inference_state,
                        frame_idx,
                        box,
                        object_id
                    )
        
        # Propagate through video
        video_segments = self.sam2.propagate_video_masks(inference_state)
        
        # Add DINOv3 guidance if requested
        if use_dinov3_guidance:
            video_segments = self._add_dinov3_guidance(frames, video_segments, initial_prompts)
        
        return {
            'segments': video_segments,
            'inference_state': inference_state,
            'num_frames': len(frames),
            'initial_prompts': initial_prompts
        }
    
    def segment_video_sequence(
        self,
        video_path: str,
        prompts: Dict[int, Dict],
        output_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_visualizations: bool = True
    ) -> Dict:
        """
        Complete video segmentation workflow
        
        Args:
            video_path: Path to input video
            prompts: Initial prompts for tracking
            output_dir: Directory to save results
            max_frames: Maximum frames to process
            save_visualizations: Whether to save visualization frames
            
        Returns:
            Complete segmentation results
        """
        # Load video frames
        frames = self.load_video_frames(video_path, max_frames)
        
        # Perform tracking with SAM 2
        tracking_results = self.track_with_sam2(frames, prompts, use_dinov3_guidance=True)
        
        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save masks and visualizations
            if save_visualizations:
                self._save_video_results(frames, tracking_results, output_dir)
        
        return tracking_results
    
    def _find_detections_in_similarity_map(
        self,
        similarity_map: np.ndarray,
        frame_shape: Tuple[int, int],
        threshold: float
    ) -> List[Dict]:
        """Find object detections in similarity map"""
        from scipy import ndimage
        
        # Threshold similarity map
        binary_mask = similarity_map > threshold
        
        if not binary_mask.any():
            return []
        
        # Find connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        detections = []
        for label_id in range(1, num_labels + 1):
            component_mask = labeled_mask == label_id
            
            if component_mask.sum() < 10:  # Minimum size
                continue
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Scale to frame coordinates
            h_frame, w_frame = frame_shape
            h_sim, w_sim = similarity_map.shape
            
            x_min_frame = int(x_min * w_frame / w_sim)
            y_min_frame = int(y_min * h_frame / h_sim)
            x_max_frame = int(x_max * w_frame / w_sim)
            y_max_frame = int(y_max * h_frame / h_sim)
            
            # Get center and score
            center_x = (x_min_frame + x_max_frame) // 2
            center_y = (y_min_frame + y_max_frame) // 2
            score = similarity_map[component_mask].mean()
            
            detections.append({
                'bbox': (x_min_frame, y_min_frame, x_max_frame - x_min_frame, y_max_frame - y_min_frame),
                'center': (center_x, center_y),
                'score': float(score),
                'area': int(component_mask.sum())
            })
        
        # Sort by score
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections
    
    def _compute_temporal_consistency(self, prev_detections: List[Dict], curr_detections: List[Dict]) -> float:
        """Compute temporal consistency between consecutive frames"""
        if not prev_detections or not curr_detections:
            return 0.0
        
        # Simple IoU-based consistency for top detection
        prev_bbox = prev_detections[0]['bbox']
        curr_bbox = curr_detections[0]['bbox']
        
        # Compute IoU
        x1 = max(prev_bbox[0], curr_bbox[0])
        y1 = max(prev_bbox[1], curr_bbox[1])
        x2 = min(prev_bbox[0] + prev_bbox[2], curr_bbox[0] + curr_bbox[2])
        y2 = min(prev_bbox[1] + prev_bbox[3], curr_bbox[1] + curr_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = prev_bbox[2] * prev_bbox[3]
        area2 = curr_bbox[2] * curr_bbox[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _track_frame_by_frame(self, frames: List[np.ndarray], initial_prompts: Dict) -> Dict:
        """Fallback frame-by-frame tracking when video predictor is not available"""
        logger.info("Using frame-by-frame SAM 2 tracking")
        
        segments = {}
        
        for frame_idx, frame in enumerate(frames):
            # Set current frame in SAM 2
            self.sam2.set_image(frame)
            
            # Get prompts for this frame or use from initial prompts
            frame_prompts = initial_prompts.get(frame_idx, initial_prompts.get(0, {}))
            
            if 'points' in frame_prompts:
                result = self.sam2.predict_from_points(
                    frame_prompts['points'],
                    frame_prompts.get('labels')
                )
                segments[frame_idx] = {
                    'object_ids': [frame_prompts.get('object_id', 1)],
                    'mask_logits': result['masks']
                }
            elif 'boxes' in frame_prompts:
                result = self.sam2.predict_from_boxes(frame_prompts['boxes'])
                segments[frame_idx] = {
                    'object_ids': [frame_prompts.get('object_id', 1)],
                    'mask_logits': result['masks']
                }
        
        return segments
    
    def _add_dinov3_guidance(self, frames: List[np.ndarray], video_segments: Dict, initial_prompts: Dict) -> Dict:
        """Add DINOv3 guidance to SAM 2 tracking results"""
        # This would implement DINOv3-based refinement of SAM 2 results
        # For now, return segments as-is
        logger.info("DINOv3 guidance would be applied here")
        return video_segments
    
    def _save_video_results(self, frames: List[np.ndarray], results: Dict, output_dir: str):
        """Save video tracking results to files"""
        output_path = Path(output_dir)
        
        # Save individual mask frames
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        # Save visualization frames
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        # This would implement saving of masks and visualizations
        # For now, just log the action
        logger.info(f"Would save {len(frames)} frames of results")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the video pipeline"""
        return {
            'dinov3_model': self.dinov3.get_model_info(),
            'sam2_model': self.sam2.get_model_info(),
            'temporal_consistency_weight': self.temporal_consistency_weight
        }