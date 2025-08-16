"""
Download script for benchmark datasets

This script downloads and sets up RefCOCO family datasets for evaluation.
"""

import os
import requests
import zipfile
import json
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger("omnivision")


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> None:
    """Download a file with progress tracking"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
    
    print()  # New line after progress


def setup_refcoco_data(data_root: str = "data/benchmarks") -> None:
    """
    Download and setup RefCOCO family datasets
    
    Args:
        data_root: Root directory to store datasets
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up RefCOCO datasets in: {data_root}")
    
    # RefCOCO dataset URLs (these would need to be actual URLs)
    # Note: Actual RefCOCO data requires registration and agreement to terms
    urls = {
        'refcoco': {
            'instances': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.json',
            'refs': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refs(refcoco).json'
        },
        'refcoco+': {
            'instances': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.json', 
            'refs': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refs(refcoco+).json'
        },
        'refcocog': {
            'instances': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.json',
            'refs': 'http://bvisionweb1.cs.unc.edu/licheng/referit/data/refs(refcocog).json'
        }
    }
    
    # COCO Images (MS COCO 2014)
    coco_urls = {
        'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
        'val2014': 'http://images.cocodataset.org/zips/val2014.zip'
    }
    
    print("=" * 60)
    print("RefCOCO Dataset Setup Instructions")
    print("=" * 60)
    print()
    print("Due to licensing requirements, you need to manually download the RefCOCO datasets.")
    print()
    print("1. Register at: http://bvisionweb1.cs.unc.edu/licheng/referit/")
    print("2. Download the following files:")
    print()
    
    for dataset, files in urls.items():
        dataset_dir = data_root / dataset
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"For {dataset.upper()}:")
        for file_type, url in files.items():
            target_file = dataset_dir / f"{file_type}.json"
            print(f"  - Download: {url}")
            print(f"    Save as: {target_file}")
        print()
    
    # Create images directory structure
    images_dir = data_root / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("3. Download COCO 2014 images:")
    for split, url in coco_urls.items():
        split_dir = images_dir / split
        split_dir.mkdir(exist_ok=True)
        print(f"  - Download: {url}")
        print(f"    Extract to: {split_dir}")
    print()
    
    # Create sample directory structure
    create_sample_structure(data_root)
    
    print("4. After downloading, your directory structure should look like:")
    print_directory_structure(data_root)


def create_sample_structure(data_root: Path) -> None:
    """Create sample directory structure and config files"""
    
    # Create sample annotation file for testing
    sample_dir = data_root / "sample"
    sample_dir.mkdir(exist_ok=True)
    
    # Sample annotation format
    sample_annotation = {
        "images": [
            {
                "id": 1,
                "file_name": "sample_image.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "sentence": "a red car on the left side",
                "bbox": [100, 100, 200, 150],
                "segmentation": {
                    "size": [480, 640],
                    "counts": "sample_rle_data"
                }
            }
        ]
    }
    
    with open(sample_dir / "sample_annotations.json", 'w') as f:
        json.dump(sample_annotation, f, indent=2)
    
    # Create config file
    config = {
        "datasets": {
            "refcoco": {
                "data_root": str(data_root),
                "splits": ["train", "val", "testA", "testB"]
            },
            "refcoco+": {
                "data_root": str(data_root),
                "splits": ["train", "val", "testA", "testB"]
            },
            "refcocog": {
                "data_root": str(data_root),
                "splits": ["train", "val", "test"]
            }
        },
        "evaluation": {
            "metrics": ["mIoU", "oIoU", "precision", "recall", "f1_score"],
            "thresholds": [0.5, 0.7, 0.9],
            "max_samples": 1000
        }
    }
    
    with open(data_root / "benchmark_config.json", 'w') as f:
        json.dump(config, f, indent=2)


def print_directory_structure(data_root: Path) -> None:
    """Print expected directory structure"""
    structure = f"""
{data_root}/
├── refcoco/
│   ├── instances.json
│   └── refs(refcoco).json
├── refcoco+/
│   ├── instances.json  
│   └── refs(refcoco+).json
├── refcocog/
│   ├── instances.json
│   └── refs(refcocog).json
├── images/
│   ├── train2014/
│   │   ├── COCO_train2014_000000000009.jpg
│   │   └── ... (82,783 images)
│   └── val2014/
│       ├── COCO_val2014_000000000042.jpg
│       └── ... (40,504 images)
├── sample/
│   └── sample_annotations.json
└── benchmark_config.json
"""
    print(structure)


def verify_dataset(data_root: str, dataset: str = "refcoco") -> bool:
    """
    Verify that a dataset is properly set up
    
    Args:
        data_root: Root directory containing datasets
        dataset: Dataset name to verify
        
    Returns:
        True if dataset is properly set up
    """
    data_root = Path(data_root)
    dataset_dir = data_root / dataset
    
    required_files = [
        dataset_dir / "instances.json",
        dataset_dir / f"refs({dataset}).json"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Missing required file: {file_path}")
            return False
        
        # Check if file is valid JSON
        try:
            with open(file_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {file_path}")
            return False
    
    # Check images directory
    images_dir = data_root / "images"
    if not images_dir.exists():
        logger.error(f"Missing images directory: {images_dir}")
        return False
    
    # Check for some images in train2014 and val2014
    for split in ['train2014', 'val2014']:
        split_dir = images_dir / split
        if not split_dir.exists():
            logger.warning(f"Missing split directory: {split_dir}")
        else:
            # Check if directory has some images
            image_files = list(split_dir.glob("*.jpg"))
            if len(image_files) == 0:
                logger.warning(f"No images found in: {split_dir}")
    
    logger.info(f"Dataset {dataset} verification passed")
    return True


def create_test_dataset(output_path: str, num_samples: int = 10) -> None:
    """
    Create a small test dataset for development
    
    Args:
        output_path: Path to save test dataset
        num_samples: Number of test samples to create
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_path / "images" 
    images_dir.mkdir(exist_ok=True)
    
    # Create test annotations
    images_data = []
    annotations_data = []
    
    for i in range(num_samples):
        # Create synthetic image
        img = Image.new('RGB', (320, 240), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        
        # Add some shapes
        if i % 3 == 0:
            # Circle
            draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0))
            description = "red circle"
        elif i % 3 == 1:
            # Rectangle  
            draw.rectangle([50, 50, 150, 150], fill=(0, 255, 0))
            description = "green rectangle"
        else:
            # Triangle (approximated with polygon)
            draw.polygon([(100, 50), (50, 150), (150, 150)], fill=(0, 0, 255))
            description = "blue triangle"
        
        # Save image
        image_file = f"test_image_{i:04d}.jpg"
        img.save(images_dir / image_file)
        
        # Add to annotations
        images_data.append({
            "id": i,
            "file_name": image_file,
            "width": 320,
            "height": 240
        })
        
        annotations_data.append({
            "id": i,
            "image_id": i,
            "sentence": description,
            "bbox": [50, 50, 100, 100],  # Simple bbox
            "area": 10000
        })
    
    # Save annotations
    test_annotations = {
        "images": images_data,
        "annotations": annotations_data
    }
    
    with open(output_path / "test_annotations.json", 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    logger.info(f"Created test dataset with {num_samples} samples at: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup benchmark datasets")
    parser.add_argument("--data_root", default="data/benchmarks", 
                       help="Root directory for datasets")
    parser.add_argument("--create_test", action="store_true",
                       help="Create test dataset for development")
    parser.add_argument("--verify", type=str, 
                       help="Verify dataset setup (dataset name)")
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_dataset(os.path.join(args.data_root, "test"))
    elif args.verify:
        verify_dataset(args.data_root, args.verify)
    else:
        setup_refcoco_data(args.data_root)