"""
Evaluation metrics for referring expression segmentation

This module implements standard metrics used in text-guided segmentation:
- Intersection over Union (IoU)
- Overall IoU (oIoU) 
- Mean IoU (mIoU)
- Precision, Recall, F1-score
- Accuracy metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
import logging

logger = logging.getLogger("omnivision")


class SegmentationMetrics:
    """
    Comprehensive metrics for segmentation evaluation
    
    Implements standard metrics used in referring expression segmentation:
    - IoU (Intersection over Union)
    - mIoU (mean IoU across all samples)
    - oIoU (overall IoU across all pixels)
    - Precision, Recall, F1-score
    - Pixel accuracy
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            threshold: Binary threshold for converting predictions to masks
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.total_intersection = 0
        self.total_union = 0
        self.ious = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.accuracies = []
        self.num_samples = 0
    
    def update(
        self, 
        pred_mask: Union[np.ndarray, torch.Tensor], 
        gt_mask: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update metrics with a single prediction
        
        Args:
            pred_mask: Predicted segmentation mask (H, W) or (H, W, 1)
            gt_mask: Ground truth mask (H, W) or (H, W, 1)
            
        Returns:
            Dictionary with metrics for this sample
        """
        # Convert to numpy if needed
        if torch.is_tensor(pred_mask):
            pred_mask = pred_mask.cpu().numpy()
        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.cpu().numpy()
        
        # Ensure 2D
        if pred_mask.ndim > 2:
            pred_mask = pred_mask.squeeze()
        if gt_mask.ndim > 2:
            gt_mask = gt_mask.squeeze()
        
        # Binarize predictions
        if pred_mask.dtype == bool:
            pred_binary = pred_mask
        else:
            pred_binary = pred_mask > self.threshold
        
        # Binarize ground truth
        if gt_mask.dtype == bool:
            gt_binary = gt_mask
        else:
            gt_binary = gt_mask > self.threshold
        
        # Compute intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Compute IoU
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        # Compute precision, recall, F1
        pred_positive = pred_binary.sum()
        gt_positive = gt_binary.sum()
        
        if pred_positive == 0:
            precision = 1.0 if gt_positive == 0 else 0.0
        else:
            precision = intersection / pred_positive
        
        if gt_positive == 0:
            recall = 1.0 if pred_positive == 0 else 0.0
        else:
            recall = intersection / gt_positive
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # Compute pixel accuracy
        correct_pixels = np.logical_not(np.logical_xor(pred_binary, gt_binary)).sum()
        total_pixels = pred_binary.size
        accuracy = correct_pixels / total_pixels
        
        # Update accumulators
        self.total_intersection += intersection
        self.total_union += union
        self.ious.append(iou)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.accuracies.append(accuracy)
        self.num_samples += 1
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics across all samples
        
        Returns:
            Dictionary with aggregated metrics
        """
        if self.num_samples == 0:
            logger.warning("No samples processed for metrics computation")
            return {}
        
        # Mean IoU (mIoU)
        miou = np.mean(self.ious)
        
        # Overall IoU (oIoU) 
        if self.total_union == 0:
            oiou = 1.0
        else:
            oiou = self.total_intersection / self.total_union
        
        # Mean metrics
        mean_precision = np.mean(self.precisions)
        mean_recall = np.mean(self.recalls)
        mean_f1 = np.mean(self.f1_scores)
        mean_accuracy = np.mean(self.accuracies)
        
        # Additional statistics
        median_iou = np.median(self.ious)
        std_iou = np.std(self.ious)
        
        return {
            'mIoU': miou,
            'oIoU': oiou,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1_score': mean_f1,
            'accuracy': mean_accuracy,
            'median_iou': median_iou,
            'std_iou': std_iou,
            'num_samples': self.num_samples
        }
    
    def compute_per_threshold(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        thresholds: List[float] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Compute metrics across multiple thresholds
        
        Args:
            pred_masks: List of predicted masks
            gt_masks: List of ground truth masks
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary mapping threshold to metrics
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        
        for threshold in thresholds:
            # Create new metrics instance with this threshold
            metrics = SegmentationMetrics(threshold=threshold)
            
            # Evaluate all samples
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                metrics.update(pred_mask, gt_mask)
            
            # Store results
            results[threshold] = metrics.compute()
        
        return results


class RefCOCOMetrics(SegmentationMetrics):
    """
    Specialized metrics for RefCOCO evaluation
    
    Follows the evaluation protocol used in RefCOCO papers
    """
    
    def __init__(self):
        super().__init__(threshold=0.5)
        self.per_category_ious = {}
    
    def update_with_category(
        self,
        pred_mask: Union[np.ndarray, torch.Tensor],
        gt_mask: Union[np.ndarray, torch.Tensor],
        category_id: int
    ) -> Dict[str, float]:
        """
        Update metrics with category information
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask  
            category_id: Object category ID
            
        Returns:
            Metrics for this sample
        """
        metrics = self.update(pred_mask, gt_mask)
        
        # Track per-category IoU
        if category_id not in self.per_category_ious:
            self.per_category_ious[category_id] = []
        self.per_category_ious[category_id].append(metrics['iou'])
        
        return metrics
    
    def compute_category_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-category metrics
        
        Returns:
            Dictionary mapping category ID to metrics
        """
        category_metrics = {}
        
        for category_id, ious in self.per_category_ious.items():
            category_metrics[category_id] = {
                'mIoU': np.mean(ious),
                'median_iou': np.median(ious),
                'std_iou': np.std(ious),
                'num_samples': len(ious)
            }
        
        return category_metrics


def compute_batch_metrics(
    pred_masks: Union[List, np.ndarray, torch.Tensor],
    gt_masks: Union[List, np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions
    
    Args:
        pred_masks: Batch of predicted masks
        gt_masks: Batch of ground truth masks
        threshold: Binary threshold
        
    Returns:
        Aggregated metrics
    """
    metrics = SegmentationMetrics(threshold=threshold)
    
    # Handle different input types
    if isinstance(pred_masks, (list, tuple)):
        for pred, gt in zip(pred_masks, gt_masks):
            metrics.update(pred, gt)
    else:
        # Assume batched tensors/arrays
        if torch.is_tensor(pred_masks):
            pred_masks = pred_masks.cpu().numpy()
        if torch.is_tensor(gt_masks):
            gt_masks = gt_masks.cpu().numpy()
        
        for i in range(pred_masks.shape[0]):
            metrics.update(pred_masks[i], gt_masks[i])
    
    return metrics.compute()


def precision_recall_curve(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve
    
    Args:
        pred_masks: List of predicted masks (continuous values)
        gt_masks: List of ground truth masks (binary)
        num_thresholds: Number of threshold points
        
    Returns:
        Tuple of (precisions, recalls, thresholds)
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        metrics = SegmentationMetrics(threshold=threshold)
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            metrics.update(pred_mask, gt_mask)
        
        results = metrics.compute()
        precisions.append(results['precision'])
        recalls.append(results['recall'])
    
    return np.array(precisions), np.array(recalls), thresholds


def average_precision(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray]
) -> float:
    """
    Compute Average Precision (AP)
    
    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        
    Returns:
        Average precision score
    """
    precisions, recalls, _ = precision_recall_curve(pred_masks, gt_masks)
    
    # Compute AP using trapezoidal rule
    ap = np.trapz(precisions, recalls)
    return ap


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metric values
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("Segmentation Metrics:")
    lines.append("-" * 30)
    
    # Key metrics first
    key_metrics = ['mIoU', 'oIoU', 'precision', 'recall', 'f1_score', 'accuracy']
    
    for metric in key_metrics:
        if metric in metrics:
            value = metrics[metric]
            lines.append(f"{metric:<12}: {value:.{precision}f}")
    
    # Additional metrics
    other_metrics = set(metrics.keys()) - set(key_metrics) - {'num_samples'}
    for metric in sorted(other_metrics):
        value = metrics[metric]
        if isinstance(value, (int, float)):
            lines.append(f"{metric:<12}: {value:.{precision}f}")
    
    if 'num_samples' in metrics:
        lines.append(f"{'samples':<12}: {metrics['num_samples']}")
    
    return '\n'.join(lines)