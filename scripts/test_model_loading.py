#!/usr/bin/env python3
"""
Test script for DINOv3 model loading on Mac

This script tests loading and basic functionality of the available DINOv3 models:
- facebook/dinov3-vits16-pretrain-lvd1689m (ViT-Small)
- facebook/dinov3-vit7b16-pretrain-lvd1689m (ViT-7B)
"""

import time
import torch
import psutil
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_model(model_name, test_image_path=None):
    """Test loading and inference for a DINOv3 model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Memory before loading
    memory_before = get_memory_usage()
    print(f"Memory before loading: {memory_before:.1f} MB")
    
    try:
        # Load model and processor
        print("Loading model...")
        start_time = time.time()
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            print("Using CPU")
        
        # For 7B model, use float16 to save memory
        dtype = torch.float16 if "7b" in model_name else torch.float32
        
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device == "cpu" else "auto"
        )
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        memory_after_load = get_memory_usage()
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"Memory after loading: {memory_after_load:.1f} MB (+{memory_after_load - memory_before:.1f} MB)")
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params / 1e6:.1f}M")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Test with dummy image if no test image provided
        if test_image_path is None:
            print("Creating dummy image for testing...")
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            test_image = dummy_image
        else:
            print(f"Loading test image: {test_image_path}")
            test_image = Image.open(test_image_path)
        
        # Preprocess
        print("Preprocessing image...")
        inputs = processor(images=test_image, return_tensors="pt")
        
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        print("Running inference...")
        start_inference = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        inference_time = time.time() - start_inference
        memory_after_inference = get_memory_usage()
        
        print(f"✅ Inference completed in {inference_time:.3f}s")
        print(f"Memory after inference: {memory_after_inference:.1f} MB")
        
        # Analyze outputs
        last_hidden = outputs.last_hidden_state
        print(f"Output shape: {last_hidden.shape}")
        
        # Extract different token types
        num_registers = getattr(model.config, 'num_register_tokens', 4)
        
        cls_token = last_hidden[:, 0, :]
        register_tokens = last_hidden[:, 1:1+num_registers, :]
        patch_tokens = last_hidden[:, 1+num_registers:, :]
        
        print(f"CLS token shape: {cls_token.shape}")
        print(f"Register tokens shape: {register_tokens.shape}")
        print(f"Patch tokens shape: {patch_tokens.shape}")
        
        # Calculate patch grid dimensions
        num_patches = patch_tokens.shape[1]
        grid_size = int(num_patches ** 0.5)
        
        print(f"Patch grid: {grid_size} x {grid_size}")
        
        return {
            'success': True,
            'load_time': load_time,
            'inference_time': inference_time,
            'memory_usage': memory_after_inference - memory_before,
            'num_params': num_params,
            'output_shape': last_hidden.shape,
            'patch_grid_size': grid_size
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    """Main testing function"""
    print("DINOv3 Model Loading Test for Mac")
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # System info
    print(f"\nSystem Info:")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Models to test
    models_to_test = [
        "facebook/dinov3-vits16-pretrain-lvd1689m",  # Your available model
        # "facebook/dinov3-vit7b16-pretrain-lvd1689m"  # Uncomment to test 7B model
    ]
    
    results = {}
    
    for model_name in models_to_test:
        result = test_model(model_name)
        results[model_name] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        short_name = model_name.split('/')[-1]
        if result['success']:
            print(f"✅ {short_name}")
            print(f"   Load time: {result['load_time']:.2f}s")
            print(f"   Inference: {result['inference_time']:.3f}s")
            print(f"   Memory: +{result['memory_usage']:.1f} MB")
            print(f"   Params: {result['num_params']/1e6:.1f}M")
        else:
            print(f"❌ {short_name}: {result['error']}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR MAC DEVELOPMENT")
    print(f"{'='*60}")
    
    print("1. For rapid development: Use ViT-Small (facebook/dinov3-vits16-pretrain-lvd1689m)")
    print("2. For production: Consider ViT-Base with careful memory management")
    print("3. For 7B model: Use with quantization (int4/int8) only")
    print("4. Enable MPS if available for faster inference")

if __name__ == "__main__":
    main()