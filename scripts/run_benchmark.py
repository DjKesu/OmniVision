#!/usr/bin/env python3
"""
Benchmark evaluation script for OmniVision

This script runs comprehensive evaluation of the OmniVision model
on standard referring expression segmentation benchmarks.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omnivision.pipelines.text_guided import TextGuidedPipeline
from benchmarks.evaluator import BenchmarkEvaluator, quick_benchmark
from benchmarks.download_data import verify_dataset, create_test_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnivision")


def main():
    parser = argparse.ArgumentParser(description="Run OmniVision benchmark evaluation")
    
    # Model configuration
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--use_improved_fusion", action="store_true", default=True,
                       help="Use Trident-inspired fusion")
    parser.add_argument("--dinov3_model", default="facebook/dinov3-vits16-pretrain-lvd1689m",
                       help="DINOv3 model name")
    parser.add_argument("--clip_model", default="ViT-B/32", 
                       help="CLIP model name")
    parser.add_argument("--sam2_model", default="tiny",
                       help="SAM2 model size")
    
    # Dataset configuration  
    parser.add_argument("--data_root", default="data/benchmarks",
                       help="Root directory containing benchmark datasets")
    parser.add_argument("--datasets", nargs="+", default=["refcoco"],
                       help="Datasets to evaluate")
    parser.add_argument("--splits", nargs="+", default=["val"],
                       help="Dataset splits to evaluate")
    
    # Evaluation configuration
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save prediction visualizations")
    parser.add_argument("--output_dir", default="benchmark_results",
                       help="Output directory for results")
    
    # Quick options
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with limited samples")
    parser.add_argument("--create_test_data", action="store_true",
                       help="Create synthetic test dataset")
    parser.add_argument("--verify_data", action="store_true",
                       help="Verify dataset setup")
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test_data:
        logger.info("Creating test dataset...")
        test_path = os.path.join(args.data_root, "test")
        create_test_dataset(test_path, num_samples=20)
        return
    
    # Verify datasets if requested
    if args.verify_data:
        logger.info("Verifying dataset setup...")
        for dataset in args.datasets:
            if verify_dataset(args.data_root, dataset):
                logger.info(f"✅ {dataset} verification passed")
            else:
                logger.error(f"❌ {dataset} verification failed")
        return
    
    # Set quick evaluation defaults
    if args.quick:
        args.max_samples = 50
        args.datasets = ["test"] if "test" in args.datasets else args.datasets[:1]
        args.splits = ["val"] if "val" in args.splits else args.splits[:1]
        logger.info("Quick evaluation mode enabled")
    
    # Initialize model
    logger.info("Loading OmniVision pipeline...")
    try:
        model = TextGuidedPipeline(
            device=args.device,
            use_improved_fusion=args.use_improved_fusion,
            dinov3_model=args.dinov3_model,
            clip_model=args.clip_model,
            sam2_model=args.sam2_model
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Run evaluation
    if args.quick and args.max_samples and args.max_samples <= 100:
        # Use quick benchmark for fast evaluation
        logger.info("Running quick benchmark...")
        results = quick_benchmark(
            model=model,
            data_root=args.data_root,
            datasets=args.datasets,
            splits=args.splits,
            max_samples=args.max_samples,
            device=args.device
        )
        
        # Print results
        print("\n" + "="*60)
        print("QUICK BENCHMARK RESULTS")
        print("="*60)
        
        for key, metrics in results.items():
            if 'error' in metrics:
                print(f"\n{key}: ERROR - {metrics['error']}")
            else:
                print(f"\n{key}:")
                print(f"  mIoU:      {metrics.get('mIoU', 0):.4f}")
                print(f"  oIoU:      {metrics.get('oIoU', 0):.4f}")
                print(f"  Precision: {metrics.get('precision', 0):.4f}")
                print(f"  Recall:    {metrics.get('recall', 0):.4f}")
                print(f"  F1-score:  {metrics.get('f1_score', 0):.4f}")
                print(f"  Samples:   {metrics.get('num_samples', 0)}")
    
    else:
        # Full evaluation
        evaluator = BenchmarkEvaluator(
            model=model,
            device=args.device,
            output_dir=args.output_dir
        )
        
        all_results = {}
        
        for dataset in args.datasets:
            for split in args.splits:
                logger.info(f"Evaluating {dataset} {split}...")
                
                try:
                    results = evaluator.evaluate_dataset(
                        dataset=dataset,
                        data_root=args.data_root,
                        split=split,
                        batch_size=args.batch_size,
                        max_samples=args.max_samples,
                        save_predictions=args.save_predictions
                    )
                    
                    key = f"{dataset}_{split}"
                    all_results[key] = results
                    
                    # Print summary
                    metrics = results['metrics']
                    print(f"\n{key} Results:")
                    print(f"  mIoU: {metrics['mIoU']:.4f}")
                    print(f"  oIoU: {metrics['oIoU']:.4f}")
                    print(f"  F1:   {metrics['f1_score']:.4f}")
                    print(f"  Samples: {results['num_samples']}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {dataset} {split}: {e}")
                    continue
        
        # Print final summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for key, results in all_results.items():
            metrics = results['metrics']
            print(f"\n{key}:")
            print(f"  mIoU:      {metrics['mIoU']:.4f}")
            print(f"  oIoU:      {metrics['oIoU']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-score:  {metrics['f1_score']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Samples:   {results['num_samples']}")
            print(f"  Time:      {metrics['eval_time']:.1f}s")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()