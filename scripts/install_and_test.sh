#!/bin/bash

# DINOv3-SAM Installation and Testing Script for Mac
# This script sets up the environment and runs initial tests

set -e  # Exit on any error

echo "🚀 DINOv3-SAM Installation and Testing"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: environment.yml not found. Please run this script from the dinov3_sam directory."
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Step 1: Download test images
echo ""
echo "📥 Step 1: Downloading test images..."
python3 download_test_images.py

# Step 2: Check conda installation
echo ""
echo "🔍 Step 2: Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Step 3: Create conda environment
echo ""
echo "🏗️  Step 3: Creating conda environment..."
if conda env list | grep -q "dinov3-sam"; then
    echo "⚠️  Environment 'dinov3-sam' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating new environment 'dinov3-sam'..."
    conda env create -f environment.yml
fi

# Step 4: Activate environment and install package
echo ""
echo "📦 Step 4: Installing package in development mode..."

# Note: We need to activate the environment for the pip install
echo "Please run the following commands manually:"
echo ""
echo "conda activate dinov3-sam"
echo "pip install -e ."
echo ""
echo "Then continue with testing:"
echo "python test_model_loading.py"

# Step 5: Provide testing instructions
echo ""
echo "🧪 Testing Instructions:"
echo "======================="
echo ""
echo "After activating the environment and installing the package, try these commands:"
echo ""
echo "1. Test model loading:"
echo "   python test_model_loading.py"
echo ""
echo "2. Test CLI:"
echo "   dinov3-sam test-model --model facebook/dinov3-vits16-pretrain-lvd1689m --verbose"
echo ""
echo "3. Test with images:"
echo "   dinov3-sam test-model --image test_images/synthetic_circles.jpg"
echo ""
echo "4. Find correspondences:"
echo "   dinov3-sam similarity --img1 test_images/synthetic_circles.jpg --img2 test_images/synthetic_squares.jpg --correspondences 5"
echo ""
echo "5. Test localization:"
echo "   dinov3-sam localize --ref test_images/synthetic_circles.jpg --click 128,128 --threshold 0.5"
echo ""
echo "6. Benchmark performance:"
echo "   dinov3-sam benchmark --model small --num-images 5"

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. conda activate dinov3-sam"
echo "2. pip install -e ."
echo "3. Run the test commands above"
echo ""
echo "If you encounter any issues:"
echo "- Check that you have sufficient RAM (8GB+ recommended)"
echo "- Try with ViT-Small model first"
echo "- Use --verbose flag for detailed error messages"

echo ""
echo "✅ Setup preparation complete!"