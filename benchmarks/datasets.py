"""
Dataset loaders for referring expression segmentation benchmarks

This module provides data loaders for standard benchmarks:
- RefCOCO, RefCOCO+, RefCOCOg
- Custom dataset support for evaluation
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("omnivision")


class RefCOCODataset(Dataset):
    """
    RefCOCO dataset for referring expression segmentation
    
    Supports RefCOCO, RefCOCO+, and RefCOCOg variants
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str = "refcoco",
        split: str = "val",
        image_size: Tuple[int, int] = (512, 512),
        transform=None
    ):
        """
        Initialize RefCOCO dataset
        
        Args:
            data_root: Root directory containing dataset files
            dataset_name: One of ["refcoco", "refcoco+", "refcocog"]
            split: Dataset split ("train", "val", "test", "testA", "testB")
            image_size: Target image size for resizing
            transform: Optional image transforms
        """
        self.data_root = data_root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Validate dataset name
        valid_datasets = ["refcoco", "refcoco+", "refcocog"]
        if self.dataset_name not in valid_datasets:
            raise ValueError(f"Dataset must be one of {valid_datasets}")
        
        # Load annotations
        self.annotations = self._load_annotations()
        logger.info(f"Loaded {len(self.annotations)} samples from {dataset_name} {split}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations from files"""
        # Expected file structure:
        # data_root/
        #   refcoco/
        #     instances.json
        #     refs(dataset).json
        #   images/
        #     train2014/
        #     val2014/
        
        dataset_dir = os.path.join(self.data_root, self.dataset_name)
        
        # Load instance annotations
        instances_file = os.path.join(dataset_dir, "instances.json") 
        if not os.path.exists(instances_file):
            raise FileNotFoundError(f"Instances file not found: {instances_file}")
        
        with open(instances_file, 'r') as f:
            instances_data = json.load(f)
        
        # Load referring expressions
        refs_file = os.path.join(dataset_dir, f"refs({self.dataset_name}).json")
        if not os.path.exists(refs_file):
            raise FileNotFoundError(f"Refs file not found: {refs_file}")
            
        with open(refs_file, 'r') as f:
            refs_data = json.load(f)
        
        # Filter by split
        refs_filtered = [ref for ref in refs_data if ref['split'] == self.split]
        
        # Create image and annotation mappings
        images = {img['id']: img for img in instances_data['images']}
        annotations = {ann['id']: ann for ann in instances_data['annotations']}
        
        # Build dataset samples
        samples = []
        for ref in refs_filtered:
            image_id = ref['image_id']
            ann_id = ref['ann_id']
            
            if image_id not in images or ann_id not in annotations:
                continue
                
            image_info = images[image_id]
            ann_info = annotations[ann_id]
            
            # Get all sentences for this referring expression
            for sentence in ref['sentences']:
                sample = {
                    'ref_id': ref['ref_id'],
                    'image_id': image_id,
                    'ann_id': ann_id,
                    'image_file': image_info['file_name'],
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'sentence': sentence['sent'],
                    'bbox': ann_info['bbox'],  # [x, y, w, h]
                    'segmentation': ann_info['segmentation'],
                    'area': ann_info['area'],
                    'category_id': ann_info['category_id']
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.annotations[idx]
        
        # Load image
        image_path = self._get_image_path(sample['image_file'])
        image = Image.open(image_path).convert('RGB')
        
        # Create segmentation mask
        mask = self._create_mask(sample, image.size)
        
        # Resize if needed
        if self.image_size and self.image_size != image.size:
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': np.array(mask),
            'sentence': sample['sentence'],
            'ref_id': sample['ref_id'],
            'image_id': sample['image_id'],
            'bbox': sample['bbox'],
            'original_size': (sample['width'], sample['height'])
        }
    
    def _get_image_path(self, filename: str) -> str:
        """Get full path to image file"""
        # Determine subdirectory based on filename
        if filename.startswith('COCO_train2014'):
            subdir = 'train2014'
        elif filename.startswith('COCO_val2014'):
            subdir = 'val2014'
        else:
            # Try to infer from filename
            if 'train' in filename:
                subdir = 'train2014'
            else:
                subdir = 'val2014'
        
        return os.path.join(self.data_root, 'images', subdir, filename)
    
    def _create_mask(self, sample: Dict, image_size: Tuple[int, int]) -> Image.Image:
        """Create binary segmentation mask from RLE or polygon"""
        from pycocotools import mask as coco_mask
        
        width, height = image_size
        segmentation = sample['segmentation']
        
        if isinstance(segmentation, list):
            # Polygon format
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
        else:
            # RLE format
            rle = segmentation
        
        # Decode to binary mask
        binary_mask = coco_mask.decode(rle)
        return Image.fromarray(binary_mask.astype(np.uint8) * 255, mode='L')


class RefCOCOPlusDataset(RefCOCODataset):
    """RefCOCO+ dataset (no location words)"""
    
    def __init__(self, *args, **kwargs):
        kwargs['dataset_name'] = 'refcoco+'
        super().__init__(*args, **kwargs)


class RefCOCOgDataset(RefCOCODataset):
    """RefCOCOg dataset (longer expressions)"""
    
    def __init__(self, *args, **kwargs):
        kwargs['dataset_name'] = 'refcocog'
        super().__init__(*args, **kwargs)


class CustomDataset(Dataset):
    """
    Custom dataset for evaluating on user-provided data
    
    Expected format:
    {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg", 
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "sentence": "red car on the left",
                "segmentation": [...],  # RLE or polygon
                "bbox": [x, y, w, h]
            }
        ]
    }
    """
    
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        transform=None
    ):
        """
        Initialize custom dataset
        
        Args:
            annotation_file: JSON file with annotations
            image_dir: Directory containing images
            image_size: Target image size
            transform: Optional transforms
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        
        logger.info(f"Loaded {len(self.annotations)} samples from custom dataset")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        ann = self.annotations[idx]
        image_info = self.images[ann['image_id']]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Create mask if segmentation provided
        mask = None
        if 'segmentation' in ann:
            mask = self._create_mask(ann, image.size)
        
        # Resize if needed
        if self.image_size and self.image_size != image.size:
            image = image.resize(self.image_size, Image.BILINEAR)
            if mask is not None:
                mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'sentence': ann['sentence'],
            'image_id': ann['image_id'],
            'original_size': (image_info['width'], image_info['height'])
        }
        
        if mask is not None:
            result['mask'] = np.array(mask)
        
        if 'bbox' in ann:
            result['bbox'] = ann['bbox']
        
        return result
    
    def _create_mask(self, ann: Dict, image_size: Tuple[int, int]) -> Image.Image:
        """Create mask from segmentation data"""
        from pycocotools import mask as coco_mask
        
        width, height = image_size
        segmentation = ann['segmentation']
        
        if isinstance(segmentation, list):
            # Polygon format
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
        else:
            # RLE format
            rle = segmentation
        
        # Decode to binary mask
        binary_mask = coco_mask.decode(rle)
        return Image.fromarray(binary_mask.astype(np.uint8) * 255, mode='L')


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create dataloader for benchmark evaluation"""
    
    def collate_fn(batch):
        """Custom collate function to handle variable-sized data"""
        return batch  # Return list of samples for easier processing
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )