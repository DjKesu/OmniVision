#!/usr/bin/env python3
"""
Download test images for OmniVision

This script downloads a variety of test images suitable for testing:
- Correspondence matching
- Object localization
- Similarity computation
"""

import os
import requests
from pathlib import Path
from PIL import Image
import numpy as np

def download_image(url, filename, test_images_dir):
    """Download image from URL"""
    filepath = test_images_dir / filename
    
    if filepath.exists():
        print(f"✅ {filename} already exists")
        return True
    
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify it's a valid image
        img = Image.open(filepath)
        img.verify()
        
        print(f"✅ Downloaded {filename} ({img.size[0]}x{img.size[1]})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def create_synthetic_images(test_images_dir):
    """Create some synthetic test images"""
    print("🎨 Creating synthetic test images...")
    
    # Simple geometric patterns for testing
    patterns = [
        ("circles", lambda: create_circles_image()),
        ("squares", lambda: create_squares_image()),
        ("gradient", lambda: create_gradient_image())
    ]
    
    for name, create_func in patterns:
        filepath = test_images_dir / f"synthetic_{name}.jpg"
        if not filepath.exists():
            try:
                img_array = create_func()
                img = Image.fromarray(img_array)
                img.save(filepath, "JPEG", quality=95)
                print(f"✅ Created synthetic_{name}.jpg")
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")

def create_circles_image(size=(512, 512)):
    """Create image with colored circles"""
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add some colored circles
    centers = [(128, 128), (384, 128), (256, 384), (128, 384), (384, 384)]
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
    
    y, x = np.ogrid[:size[0], :size[1]]
    
    for (cx, cy), color in zip(centers, colors):
        mask = (x - cx)**2 + (y - cy)**2 <= 40**2
        img[mask] = color
    
    return img

def create_squares_image(size=(512, 512)):
    """Create image with colored squares"""
    img = np.ones((*size, 3), dtype=np.uint8) * 240
    
    # Add colored squares
    squares = [
        (50, 50, 100, (255, 0, 0)),
        (200, 50, 80, (0, 255, 0)),
        (400, 50, 60, (0, 0, 255)),
        (50, 200, 120, (255, 255, 0)),
        (300, 300, 100, (255, 0, 255))
    ]
    
    for x, y, size_sq, color in squares:
        img[y:y+size_sq, x:x+size_sq] = color
    
    return img

def create_gradient_image(size=(512, 512)):
    """Create gradient image"""
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    # Create RGB gradients
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j, 0] = int(255 * i / size[0])  # Red gradient
            img[i, j, 1] = int(255 * j / size[1])  # Green gradient
            img[i, j, 2] = int(255 * (i + j) / (size[0] + size[1]))  # Blue gradient
    
    return img

def main():
    """Main function to download test images"""
    
    # Create test images directory
    test_images_dir = Path("test_images")
    test_images_dir.mkdir(exist_ok=True)
    
    print(f"📁 Test images directory: {test_images_dir.absolute()}")
    
    # Sample images from various sources (using royalty-free/public domain images)
    image_urls = [
        # Wikimedia Commons - Public domain images
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png", "lenna.png"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Cornell_box.png/256px-Cornell_box.png", "cornell_box.png"),
        
        # Unsplash sample images (small sizes, should be OK for testing)
        ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop", "mountain.jpg"),
        ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=300&fit=crop", "cat.jpg"),
        ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=300&fit=crop", "dog.jpg"),
        ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&crop=top", "mountain_cropped.jpg"),
    ]
    
    print("🌐 Downloading test images...")
    
    successful_downloads = 0
    
    for url, filename in image_urls:
        if download_image(url, filename, test_images_dir):
            successful_downloads += 1
    
    print(f"\n📊 Downloaded {successful_downloads}/{len(image_urls)} images successfully")
    
    # Create synthetic images as backup
    create_synthetic_images(test_images_dir)
    
    # List all available test images
    print(f"\n📋 Available test images:")
    image_files = list(test_images_dir.glob("*"))
    image_files.sort()
    
    for img_file in image_files:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img = Image.open(img_file)
                print(f"  ✅ {img_file.name} ({img.size[0]}x{img.size[1]})")
            except Exception:
                print(f"  ❌ {img_file.name} (corrupted)")
    
    if not image_files:
        print("  ⚠️  No images found. Creating more synthetic images...")
        
        # Create additional synthetic images
        additional_patterns = [
            ("checkerboard", create_checkerboard),
            ("stripes", create_stripes),
            ("noise", create_noise_image)
        ]
        
        for name, create_func in additional_patterns:
            filepath = test_images_dir / f"synthetic_{name}.jpg"
            try:
                img_array = create_func()
                img = Image.fromarray(img_array)
                img.save(filepath, "JPEG", quality=95)
                print(f"  ✅ Created synthetic_{name}.jpg")
            except Exception as e:
                print(f"  ❌ Failed to create {name}: {e}")
    
    print(f"\n🎯 Test images ready in: {test_images_dir.absolute()}")
    print("\nNext steps:")
    print("1. conda env create -f environment.yml")
    print("2. conda activate dinov3-sam")
    print("3. pip install -e .")
    print("4. python test_model_loading.py")

def create_checkerboard(size=(512, 512), square_size=32):
    """Create checkerboard pattern"""
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    for i in range(0, size[0], square_size):
        for j in range(0, size[1], square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255
    
    return img

def create_stripes(size=(512, 512), stripe_width=20):
    """Create stripe pattern"""
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    for i in range(size[1]):
        if (i // stripe_width) % 2 == 0:
            img[:, i] = [255, 100, 100]  # Red stripes
        else:
            img[:, i] = [100, 100, 255]  # Blue stripes
    
    return img

def create_noise_image(size=(512, 512)):
    """Create random noise image"""
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

if __name__ == "__main__":
    main()