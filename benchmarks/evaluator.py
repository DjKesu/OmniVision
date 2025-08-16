"""
Benchmark evaluation pipeline for OmniVision

This module provides a complete evaluation framework for testing
text-guided segmentation models on standard benchmarks.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from .datasets import RefCOCODataset, RefCOCOPlusDataset, RefCOCOgDataset, CustomDataset
from .metrics import SegmentationMetrics, RefCOCOMetrics, format_metrics

logger = logging.getLogger("omnivision")


class BenchmarkEvaluator:
    """
    Comprehensive evaluation pipeline for text-guided segmentation
    
    Supports evaluation on RefCOCO family datasets and custom datasets
    with detailed metrics and analysis.
    """
    
    def __init__(
        self,
        model=None,
        device: str = 'cpu',
        output_dir: str = 'benchmark_results'
    ):
        """
        Initialize evaluator
        
        Args:
            model: Text-guided segmentation model to evaluate
            device: Device for computation
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging for results
        log_file = os.path.join(output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Benchmark evaluator initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Output directory: {output_dir}")
    
    def evaluate_dataset(
        self,
        dataset: Union[str, torch.utils.data.Dataset],
        data_root: str = None,
        split: str = 'val',
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset
        
        Args:
            dataset: Dataset name ('refcoco', 'refcoco+', 'refcocog') or Dataset object
            data_root: Root directory for dataset files (if dataset is string)
            split: Dataset split to evaluate
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate (for quick testing)
            save_predictions: Whether to save prediction visualizations
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation on {dataset} {split}")
        
        # Load dataset
        if isinstance(dataset, str):
            if data_root is None:
                raise ValueError("data_root must be provided when dataset is a string")
            
            dataset_cls = {
                'refcoco': RefCOCODataset,
                'refcoco+': RefCOCOPlusDataset,
                'refcocog': RefCOCOgDataset
            }[dataset.lower()]
            
            dataset_obj = dataset_cls(
                data_root=data_root,
                split=split,
                image_size=(512, 512)  # Standard size for evaluation
            )
            dataset_name = dataset
        else:
            dataset_obj = dataset
            dataset_name = dataset.__class__.__name__
        
        # Create dataloader
        dataloader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x  # Return list of samples
        )
        
        # Initialize metrics
        if 'refcoco' in dataset_name.lower():
            metrics = RefCOCOMetrics()
        else:
            metrics = SegmentationMetrics()
        
        # Evaluation loop
        all_predictions = []
        start_time = time.time()
        num_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
                
                # Process each sample in batch
                for sample in batch:
                    if max_samples and num_processed >= max_samples:
                        break
                    
                    # Extract data
                    image = sample['image']
                    sentence = sample['sentence']
                    gt_mask = sample.get('mask')
                    
                    if gt_mask is None:
                        logger.warning(f"No ground truth mask for sample {num_processed}")
                        continue
                    
                    # Run model prediction
                    try:
                        pred_result = self._predict_sample(image, sentence)
                        pred_mask = pred_result.get('mask')
                        
                        if pred_mask is None:
                            logger.warning(f"No prediction for sample {num_processed}")
                            continue
                        
                        # Update metrics
                        if hasattr(metrics, 'update_with_category') and 'category_id' in sample:
                            sample_metrics = metrics.update_with_category(
                                pred_mask, gt_mask, sample['category_id']
                            )
                        else:
                            sample_metrics = metrics.update(pred_mask, gt_mask)
                        
                        # Store prediction info
                        pred_info = {
                            'sample_id': num_processed,
                            'ref_id': sample.get('ref_id'),
                            'image_id': sample.get('image_id'),
                            'sentence': sentence,
                            'metrics': sample_metrics,
                            'prediction_time': pred_result.get('time', 0)
                        }
                        
                        if save_predictions:
                            pred_info['pred_mask'] = pred_mask
                            pred_info['gt_mask'] = gt_mask
                            pred_info['image'] = image
                        
                        all_predictions.append(pred_info)
                        
                    except Exception as e:
                        logger.error(f"Error processing sample {num_processed}: {e}")
                        continue
                    
                    num_processed += 1
                
                if max_samples and num_processed >= max_samples:
                    break
        
        eval_time = time.time() - start_time
        
        # Compute final metrics
        final_metrics = metrics.compute()
        
        # Add timing info
        final_metrics['eval_time'] = eval_time
        final_metrics['samples_per_second'] = num_processed / eval_time if eval_time > 0 else 0
        
        # Save detailed results
        results = {
            'dataset': dataset_name,
            'split': split,
            'num_samples': num_processed,
            'metrics': final_metrics,
            'predictions': all_predictions if save_predictions else None
        }
        
        # Add per-category metrics if available
        if hasattr(metrics, 'compute_category_metrics'):
            results['category_metrics'] = metrics.compute_category_metrics()
        
        # Save results
        result_file = os.path.join(
            self.output_dir, 
            f'{dataset_name}_{split}_results.json'
        )
        
        # Prepare JSON-serializable results
        json_results = self._prepare_for_json(results)
        with open(result_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation completed: {num_processed} samples in {eval_time:.2f}s")
        logger.info(f"Results saved to: {result_file}")
        logger.info(f"Metrics:\\n{format_metrics(final_metrics)}")
        
        # Save prediction visualizations if requested
        if save_predictions:
            self._save_prediction_visualizations(all_predictions, dataset_name, split)
        
        return results
    
    def _predict_sample(self, image: Image.Image, sentence: str) -> Dict[str, Any]:
        """
        Run model prediction on a single sample
        
        Args:
            image: Input PIL Image
            sentence: Text description
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model provided for evaluation")
        
        start_time = time.time()
        
        # Run prediction
        if hasattr(self.model, 'segment_by_text'):
            # OmniVision pipeline
            result = self.model.segment_by_text(
                image, 
                sentence,
                similarity_threshold=0.5,
                sam2_refine=True
            )
            
            # Extract best mask
            masks = result.get('masks', [])
            if len(masks) > 0:
                pred_mask = masks[0]  # Use best mask
            else:
                # Create empty mask if no prediction
                pred_mask = np.zeros((image.height, image.width), dtype=bool)
        
        else:
            # Generic model interface
            pred_mask = self.model(image, sentence)
        
        pred_time = time.time() - start_time
        
        return {
            'mask': pred_mask,
            'time': pred_time,
            'raw_result': result if 'result' in locals() else None
        }
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, (Image.Image, torch.Tensor)):
            return None  # Skip non-serializable objects
        else:
            return data
    
    def _save_prediction_visualizations(
        self, 
        predictions: List[Dict], 
        dataset_name: str, 
        split: str,
        max_visualizations: int = 50
    ):
        """Save visualization of predictions"""
        import matplotlib.pyplot as plt
        
        vis_dir = os.path.join(self.output_dir, f'{dataset_name}_{split}_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Select samples to visualize
        viz_samples = predictions[:max_visualizations]
        
        for i, pred in enumerate(viz_samples):
            if pred.get('pred_mask') is None or pred.get('gt_mask') is None:
                continue
            
            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(pred['image'])
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Ground truth
                axes[1].imshow(pred['image'])
                gt_mask = pred['gt_mask'].astype(bool)
                colored_gt = np.zeros((*gt_mask.shape, 4))
                colored_gt[gt_mask] = [0, 1, 0, 0.6]  # Green
                axes[1].imshow(colored_gt)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Prediction
                axes[2].imshow(pred['image'])
                pred_mask = pred['pred_mask'].astype(bool)
                colored_pred = np.zeros((*pred_mask.shape, 4))
                colored_pred[pred_mask] = [1, 0, 0, 0.6]  # Red
                axes[2].imshow(colored_pred)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                # Add sentence and metrics as title
                iou = pred['metrics']['iou']
                plt.suptitle(f"'{pred['sentence']}' | IoU: {iou:.3f}", fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'sample_{i:04d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.warning(f"Failed to create visualization for sample {i}: {e}")
                continue
        
        logger.info(f"Saved {len(viz_samples)} visualizations to {vis_dir}")
    
    def compare_models(
        self,
        models: Dict[str, Any],
        dataset: str,
        data_root: str,
        split: str = 'val',
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset
        
        Args:
            models: Dictionary mapping model names to model objects
            dataset: Dataset name
            data_root: Dataset root directory
            split: Dataset split
            max_samples: Maximum samples to evaluate
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models on {dataset} {split}")
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Create evaluator for this model
            evaluator = BenchmarkEvaluator(
                model=model,
                device=self.device,
                output_dir=os.path.join(self.output_dir, model_name)
            )
            
            # Run evaluation
            results = evaluator.evaluate_dataset(
                dataset=dataset,
                data_root=data_root,
                split=split,
                max_samples=max_samples
            )
            
            all_results[model_name] = results
        
        # Create comparison summary
        comparison = self._create_comparison_summary(all_results)
        
        # Save comparison results
        comparison_file = os.path.join(self.output_dir, f'model_comparison_{dataset}_{split}.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Model comparison saved to: {comparison_file}")
        
        return comparison
    
    def _create_comparison_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary comparing multiple models"""
        
        # Extract key metrics for each model
        summary = {
            'models': {},
            'rankings': {}
        }
        
        metric_names = ['mIoU', 'oIoU', 'precision', 'recall', 'f1_score', 'accuracy']
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            summary['models'][model_name] = {
                metric: metrics.get(metric, 0) for metric in metric_names
            }
            summary['models'][model_name]['num_samples'] = model_results['num_samples']
            summary['models'][model_name]['eval_time'] = metrics.get('eval_time', 0)
            summary['models'][model_name]['samples_per_second'] = metrics.get('samples_per_second', 0)
        
        # Create rankings for each metric
        for metric in metric_names:
            metric_values = [(name, summary['models'][name][metric]) 
                           for name in summary['models']]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            summary['rankings'][metric] = metric_values
        
        return summary


def quick_benchmark(
    model,
    data_root: str,
    datasets: List[str] = ['refcoco'],
    splits: List[str] = ['val'],
    max_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Quick benchmark evaluation for development
    
    Args:
        model: Model to evaluate
        data_root: Root directory for datasets
        datasets: List of datasets to evaluate
        splits: List of splits to evaluate
        max_samples: Maximum samples per dataset/split
        device: Device for computation
        
    Returns:
        Summary of results
    """
    evaluator = BenchmarkEvaluator(model=model, device=device)
    
    all_results = {}
    
    for dataset in datasets:
        for split in splits:
            key = f"{dataset}_{split}"
            
            try:
                results = evaluator.evaluate_dataset(
                    dataset=dataset,
                    data_root=data_root,
                    split=split,
                    max_samples=max_samples
                )
                all_results[key] = results['metrics']
                
            except Exception as e:
                logger.error(f"Failed to evaluate {key}: {e}")
                all_results[key] = {'error': str(e)}
    
    return all_results