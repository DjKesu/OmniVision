"""
Visualization utilities for OmniVision
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional

from .image import overlay_heatmap, draw_bbox, draw_point


def visualize_correspondences(
    image1: Union[str, Image.Image],
    image2: Union[str, Image.Image],
    matches: List[Dict],
    max_matches: int = 20,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Visualize correspondences between two images
    
    Args:
        image1: First image
        image2: Second image
        matches: List of match dictionaries
        max_matches: Maximum number of matches to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Load images
    if isinstance(image1, str):
        img1 = Image.open(image1).convert('RGB')
    else:
        img1 = image1
        
    if isinstance(image2, str):
        img2 = Image.open(image2).convert('RGB')
    else:
        img2 = image2
    
    # Convert to arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display images
    ax1.imshow(img1_array)
    ax1.set_title('Reference Image')
    ax1.axis('off')
    
    ax2.imshow(img2_array)
    ax2.set_title('Target Image')
    ax2.axis('off')
    
    # Plot correspondences
    colors = plt.cm.rainbow(np.linspace(0, 1, min(len(matches), max_matches)))
    
    for i, (match, color) in enumerate(zip(matches[:max_matches], colors)):
        coord1 = match['coord1']
        coord2 = match['coord2']
        sim = match['similarity']
        
        # Draw points
        ax1.plot(coord1[0], coord1[1], 'o', color=color, markersize=8)
        ax2.plot(coord2[0], coord2[1], 'o', color=color, markersize=8)
        
        # Add labels
        ax1.annotate(f'{i+1}', coord1, xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', weight='bold')
        ax2.annotate(f'{i+1}', coord2, xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', weight='bold')
    
    plt.tight_layout()
    return fig


def visualize_localization(
    image: Union[str, Image.Image],
    result: Dict,
    show_heatmap: bool = True,
    show_detections: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize localization results
    
    Args:
        image: Input image
        result: Localization result dictionary
        show_heatmap: Whether to show similarity heatmap
        show_detections: Whether to show detections
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image
    
    img_array = np.array(img)
    
    # Create subplots
    if show_heatmap and show_detections:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    elif show_heatmap or show_detections:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2/3, figsize[1]))
        ax3 = None
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 3, figsize[1]))
        ax2 = ax3 = None
    
    # Original image with click point
    ax1.imshow(img_array)
    ax1.set_title('Reference Image')
    ax1.axis('off')
    
    # Draw click point if available
    if 'reference_coords' in result:
        click_coords = result['reference_coords']
        ax1.plot(click_coords[0], click_coords[1], 'r*', markersize=15, label='Click')
        ax1.legend()
    
    # Similarity heatmap
    if show_heatmap and ax2 is not None:
        similarity_map = result['similarity_map'].cpu().numpy()
        overlayed = overlay_heatmap(img_array, similarity_map, alpha=0.6)
        
        ax2.imshow(overlayed)
        ax2.set_title('Similarity Heatmap')
        ax2.axis('off')
    
    # Detections
    if show_detections:
        target_ax = ax3 if ax3 is not None else (ax2 if ax2 is not None else ax1)
        
        target_ax.imshow(img_array)
        target_ax.set_title('Detections')
        target_ax.axis('off')
        
        # Draw bounding boxes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(result['detections'])))
        
        for i, (detection, color) in enumerate(zip(result['detections'], colors)):
            bbox = detection['bbox']
            sim = detection['similarity']
            
            # Create rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2], bbox[3],
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            
            target_ax.add_patch(rect)
            
            # Add label
            target_ax.annotate(
                f'{i+1}: {sim:.2f}',
                (bbox[0], bbox[1]),
                xytext=(5, -15),
                textcoords='offset points',
                fontsize=8,
                color=color,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
    
    plt.tight_layout()
    return fig


def visualize_similarity_map(
    similarity_map: Union[np.ndarray, str],
    title: str = "Similarity Map",
    colormap: str = 'hot',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Visualize similarity map
    
    Args:
        similarity_map: Similarity map array or path
        title: Plot title
        colormap: Colormap to use
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(similarity_map, str):
        similarity_map = np.load(similarity_map)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity_map, cmap=colormap)
    ax.set_title(title)
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    return fig


def plot_correspondences(
    matches: List[Dict],
    title: str = "Correspondences",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot correspondence statistics
    
    Args:
        matches: List of correspondence dictionaries
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not matches:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No correspondences found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Extract similarity scores
    similarities = [match['similarity'] for match in matches]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Similarity distribution
    ax1.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Similarity Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Similarity vs rank
    ranks = list(range(1, len(similarities) + 1))
    ax2.plot(ranks, similarities, 'bo-', markersize=4)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Similarity Score')
    ax2.set_title('Similarity vs Rank')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def create_feature_visualization(
    features: np.ndarray,
    feature_type: str = "patches",
    num_components: int = 3,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Visualize features using PCA
    
    Args:
        features: Feature array
        feature_type: Type of features
        num_components: Number of PCA components to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    # Reshape if needed
    if features.ndim > 2:
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
    else:
        original_shape = None
        features_flat = features
    
    # Apply PCA
    pca = PCA(n_components=min(num_components, features_flat.shape[1]))
    features_pca = pca.fit_transform(features_flat)
    
    # Reshape back if needed
    if original_shape is not None and len(original_shape) == 3:
        # Spatial features
        h, w = original_shape[:2]
        features_pca = features_pca.reshape(h, w, -1)
        
        fig, axes = plt.subplots(1, num_components, figsize=figsize)
        if num_components == 1:
            axes = [axes]
        
        for i in range(num_components):
            axes[i].imshow(features_pca[:, :, i], cmap='viridis')
            axes[i].set_title(f'PC {i+1} ({pca.explained_variance_ratio_[i]:.2%})')
            axes[i].axis('off')
    else:
        # Non-spatial features
        fig, ax = plt.subplots(figsize=figsize)
        
        if features_pca.shape[1] >= 2:
            scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                               c=range(len(features_pca)), cmap='viridis', alpha=0.7)
            ax.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.plot(features_pca[:, 0], 'o-')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.2%})')
    
    fig.suptitle(f'{feature_type.title()} Feature Visualization')
    plt.tight_layout()
    return fig


def save_visualization(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 150,
    bbox_inches: str = 'tight'
):
    """
    Save visualization figure
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)