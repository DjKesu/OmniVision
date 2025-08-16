"""
Gradio Interactive Demo for OmniVision

This module provides an interactive web interface for the OmniVision pipeline,
allowing users to test image segmentation, video tracking, and text-guided segmentation
through a user-friendly web interface.
"""

import gradio as gr
import numpy as np
from PIL import Image
import tempfile
import os
from typing import List, Tuple, Optional, Dict, Any
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnivision")

try:
    from ..models.dinov3_backbone import DINOv3Backbone
    from ..models.sam2_wrapper import SAM2Wrapper
    from ..models.clip_wrapper import CLIPWrapper
    from ..pipelines.segmentation import SegmentationPipeline
    from ..pipelines.video import VideoPipeline
    from ..pipelines.text_guided import TextGuidedPipeline
    from ..pipelines.similarity import SimilarityPipeline
    
    MODELS_AVAILABLE = True
    logger.info("All models imported successfully")
    
except ImportError as e:
    logger.error(f"Model import failed: {e}")
    MODELS_AVAILABLE = False

# Global model instances (initialized lazily)
_dinov3_model = None
_segmentation_pipeline = None
_text_pipeline = None
_video_pipeline = None
_similarity_pipeline = None


def get_dinov3_model():
    """Get or create DINOv3 model instance"""
    global _dinov3_model
    if _dinov3_model is None:
        _dinov3_model = DINOv3Backbone()
        logger.info("DINOv3 model loaded")
    return _dinov3_model


def get_segmentation_pipeline():
    """Get or create segmentation pipeline"""
    global _segmentation_pipeline
    if _segmentation_pipeline is None:
        _segmentation_pipeline = SegmentationPipeline()
        logger.info("Segmentation pipeline loaded")
    return _segmentation_pipeline


def get_text_pipeline():
    """Get or create text-guided pipeline"""
    global _text_pipeline
    if _text_pipeline is None:
        logger.info("Loading text-guided pipeline with improved fusion (this may take a moment)...")
        _text_pipeline = TextGuidedPipeline(
            device='cpu',  # Use CPU for demo stability
            use_improved_fusion=True,  # Enable improved cross-modal fusion
            dinov3_model="facebook/dinov3-vits16-pretrain-lvd1689m",  # Use smaller model
            clip_model="ViT-B/32",  # Use standard CLIP model
            sam2_model="tiny"  # Use smallest SAM 2 model
        )
        logger.info("Text-guided pipeline loaded successfully")
    return _text_pipeline


def get_video_pipeline():
    """Get or create video pipeline"""
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = VideoPipeline()
        logger.info("Video pipeline loaded")
    return _video_pipeline


def get_similarity_pipeline():
    """Get or create similarity pipeline"""
    global _similarity_pipeline
    if _similarity_pipeline is None:
        _similarity_pipeline = SimilarityPipeline()
        logger.info("Similarity pipeline loaded")
    return _similarity_pipeline


def click_segment(image: Image.Image, click_coords: Tuple[int, int], use_sam2: bool = True) -> Tuple[Image.Image, str]:
    """
    Segment image based on click coordinates
    
    Args:
        image: Input PIL Image
        click_coords: (x, y) coordinates of click
        use_sam2: Whether to use SAM 2 refinement
        
    Returns:
        Tuple of (result_image, status_message)
    """
    if not MODELS_AVAILABLE:
        return image, "❌ Models not available"
    
    try:
        pipeline = get_segmentation_pipeline()
        
        # Perform segmentation
        result = pipeline.segment_by_click(
            image,
            click_coords,
            sam2_refine=use_sam2
        )
        
        # Create visualization
        vis_image = pipeline.visualize_segmentation(
            image,
            result,
            show_similarity=True,
            show_masks=True
        )
        
        masks = result.get('masks', [])
        num_masks = len(masks) if isinstance(masks, list) else 0
        
        status = f"✅ Found {num_masks} mask(s) at click {click_coords}"
        
        return Image.fromarray(vis_image), status
        
    except Exception as e:
        logger.error(f"Click segmentation failed: {e}")
        return image, f"❌ Error: {str(e)}"


def text_segment(image: Image.Image, text_query: str, threshold: float = 0.3) -> Tuple[Image.Image, str]:
    """
    Segment image based on text description
    
    Args:
        image: Input PIL Image
        text_query: Text description of object to segment
        threshold: Similarity threshold
        
    Returns:
        Tuple of (result_image, status_message)
    """
    if not MODELS_AVAILABLE:
        return image, "❌ Models not available"
    
    try:
        logger.info(f"🔍 DEBUG: Starting text segmentation for query: '{text_query}'")
        logger.info(f"🔍 DEBUG: Input image type: {type(image)}")
        logger.info(f"🔍 DEBUG: Threshold: {threshold}")
        
        # Debug Gradio image format more thoroughly
        if hasattr(image, 'shape'):
            logger.info(f"🔍 DEBUG: Image shape: {image.shape}")
        if hasattr(image, 'dtype'):
            logger.info(f"🔍 DEBUG: Image dtype: {image.dtype}")
        if hasattr(image, 'size'):
            logger.info(f"🔍 DEBUG: Image size: {image.size}")
        if hasattr(image, 'mode'):
            logger.info(f"🔍 DEBUG: Image mode: {image.mode}")
        
        pipeline = get_text_pipeline()
        logger.info(f"🔍 DEBUG: Pipeline loaded successfully")
        
        # More robust image format handling
        if isinstance(image, np.ndarray):
            logger.info(f"🔍 DEBUG: Converting numpy array to PIL Image")
            logger.info(f"🔍 DEBUG: Numpy array shape: {image.shape}, dtype: {image.dtype}")
            # Handle different numpy array formats
            if image.dtype != np.uint8:
                logger.info(f"🔍 DEBUG: Converting dtype from {image.dtype} to uint8")
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            logger.info(f"🔍 DEBUG: Unknown image type, attempting conversion")
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except Exception as conv_error:
                logger.error(f"🔍 DEBUG: Image conversion failed: {conv_error}")
                raise
        else:
            logger.info(f"🔍 DEBUG: Converting PIL Image to RGB")
            image = image.convert('RGB')
        
        logger.info(f"🔍 DEBUG: Final image - size: {image.size}, mode: {image.mode}")
        
        # Perform text-guided segmentation
        logger.info(f"🔍 DEBUG: Starting pipeline.segment_by_text...")
        result = pipeline.segment_by_text(
            image,
            text_query,
            similarity_threshold=threshold,
            sam2_refine=True
        )
        logger.info(f"🔍 DEBUG: Pipeline segmentation completed")
        
        # Create better visualization
        logger.info(f"🔍 DEBUG: Starting visualization creation...")
        vis_image = create_text_segmentation_visualization(image, result)
        logger.info(f"🔍 DEBUG: Visualization completed")
        
        masks = result.get('masks', [])
        num_masks = len(masks) if isinstance(masks, list) else 0
        
        sim_info = ""
        if 'combined_similarity' in result:
            sim_map = result['combined_similarity']
            sim_range = f"[{sim_map.min():.3f}, {sim_map.max():.3f}]"
            sim_mean = f"{sim_map.mean():.3f}"
            sim_info = f", similarity: {sim_range} (μ={sim_mean})"
        
        status = f"✅ '{text_query}': {num_masks} mask(s){sim_info} | Trident-Inspired Fusion"
        
        return Image.fromarray(vis_image), status
        
    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        return image, f"❌ Error: {str(e)}"


def compute_similarity(image1: Image.Image, image2: Image.Image) -> Tuple[str, Image.Image]:
    """
    Compute similarity between two images
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        
    Returns:
        Tuple of (similarity_score, visualization_image)
    """
    if not MODELS_AVAILABLE:
        return "❌ Models not available", None
    
    try:
        pipeline = get_similarity_pipeline()
        
        # Compute overall similarity
        similarity = pipeline.compute_image_similarity(image1, image2, method="cls")
        
        # Find correspondences
        matches = pipeline.find_correspondences(image1, image2, num_matches=10)
        
        # Create visualization
        from ..utils.visualization import visualize_correspondences
        fig = visualize_correspondences(image1, image2, matches)
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis_image = Image.fromarray(vis_array)
        
        status = f"✅ Similarity: {similarity:.3f}, Found {len(matches)} correspondences"
        
        return status, vis_image
        
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        return f"❌ Error: {str(e)}", None


def text_search(image: Image.Image, queries: str) -> Tuple[str, str]:
    """
    Search image for multiple text queries
    
    Args:
        image: Input PIL Image
        queries: Comma-separated text queries
        
    Returns:
        Tuple of (rankings_text, detailed_results)
    """
    if not MODELS_AVAILABLE:
        return "❌ Models not available", ""
    
    try:
        pipeline = get_text_pipeline()
        
        # Parse queries
        query_list = [q.strip() for q in queries.split(',')]
        
        # Perform search
        result = pipeline.search_and_segment(
            image,
            query_list,
            top_k=len(query_list),
            segment_top=False  # Just ranking for now
        )
        
        # Format rankings
        rankings_text = "🔍 Query Rankings:\n"
        for i, query_info in enumerate(result['query_rankings']):
            score = query_info['overall_score']
            query = query_info['query']
            rankings_text += f"{i+1}. '{query}' ({score:.3f})\n"
        
        # Detailed results
        detailed_text = f"Searched {len(query_list)} queries:\n"
        detailed_text += f"Top match: '{result['query_rankings'][0]['query']}'\n"
        detailed_text += f"Score range: {result['query_rankings'][-1]['overall_score']:.3f} - {result['query_rankings'][0]['overall_score']:.3f}"
        
        return rankings_text, detailed_text
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return f"❌ Error: {str(e)}", ""


def create_text_segmentation_visualization(image: Image.Image, result: Dict) -> Image.Image:
    """Create proper visualization for text segmentation results"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Get results
    masks = result.get('masks', [])
    similarity_map = result.get('combined_similarity')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Similarity heatmap
    if similarity_map is not None:
        im = axes[1].imshow(similarity_map, cmap='hot', alpha=0.7)
        axes[1].imshow(image, alpha=0.3)  # Overlay original image
        axes[1].set_title('Similarity Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(image)
        axes[1].set_title('No Similarity Map')
        axes[1].axis('off')
    
    # Segmentation result
    axes[2].imshow(image)
    if len(masks) > 0:
        # Show best mask (first one, typically highest score)
        mask = masks[0].astype(bool)  # Convert to boolean for indexing
        # Create colored overlay for the mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = [1, 0, 0, 0.6]  # Red with transparency
        axes[2].imshow(colored_mask)
        axes[2].set_title(f'Segmentation ({len(masks)} masks)')
        
        # Add bounding boxes if available
        boxes = result.get('boxes', [])
        if boxes:
            for box in boxes[:3]:  # Show up to 3 boxes
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='yellow', facecolor='none')
                axes[2].add_patch(rect)
    else:
        axes[2].set_title('No Masks Found')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    try:
        # Try newer matplotlib API first
        vis_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Remove alpha
    except AttributeError:
        try:
            # Try older matplotlib API
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback: save to bytes and reload
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plt.close(fig)
            return Image.open(buf)
    
    plt.close(fig)  # Clean up memory
    return Image.fromarray(vis_array)


# Gradio interface components
def create_click_interface():
    """Create click-based segmentation interface"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="pil"
            )
            use_sam2 = gr.Checkbox(
                value=True,
                label="Use SAM 2 refinement"
            )
            segment_btn = gr.Button("Click to Segment", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Segmentation Result")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    # Handle click events
    def on_image_click(image, use_sam2, evt: gr.SelectData):
        if image is None:
            return image, "Please upload an image first"
        
        click_coords = (evt.index[0], evt.index[1])
        return click_segment(image, click_coords, use_sam2)
    
    input_image.select(
        fn=on_image_click,
        inputs=[input_image, use_sam2],
        outputs=[output_image, status_text]
    )
    
    return input_image, output_image, status_text


def create_text_interface():
    """Create text-guided segmentation interface"""
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="📷 Upload Image",
                type="pil"
            )
            text_query = gr.Textbox(
                label="📝 Describe what to segment",
                placeholder="e.g., 'bottle', 'red car', 'person', 'cat', 'drink'",
                lines=2
            )
            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.85,  # Higher default for Trident fusion
                step=0.05,
                label="🎚️ Similarity Threshold"
            )
            text_segment_btn = gr.Button("🔍 Segment by Text", variant="primary", size="lg")
            
            # Add example suggestions
            gr.Markdown("""
            **💡 Try these examples:**
            - Simple objects: "bottle", "car", "person" 
            - With colors: "red bottle", "blue car"
            - Animals: "cat", "dog", "bird"
            """)
            
        with gr.Column():
            text_output_image = gr.Image(label="🎭 Segmentation Result")
            text_status = gr.Textbox(
                label="📊 Results & Status", 
                interactive=False,
                lines=3
            )
    
    text_segment_btn.click(
        fn=text_segment,
        inputs=[input_image, text_query, threshold],
        outputs=[text_output_image, text_status]
    )
    
    return input_image, text_query, text_output_image, text_status


def create_similarity_interface():
    """Create image similarity interface"""
    with gr.Row():
        with gr.Column():
            image1 = gr.Image(label="First Image", type="pil")
            image2 = gr.Image(label="Second Image", type="pil")
            similarity_btn = gr.Button("Compute Similarity", variant="primary")
            
        with gr.Column():
            similarity_result = gr.Image(label="Similarity Visualization")
            similarity_status = gr.Textbox(label="Similarity Score", interactive=False)
    
    similarity_btn.click(
        fn=compute_similarity,
        inputs=[image1, image2],
        outputs=[similarity_status, similarity_result]
    )
    
    return image1, image2, similarity_result, similarity_status


def create_search_interface():
    """Create text search interface"""
    with gr.Row():
        with gr.Column():
            search_image = gr.Image(label="Upload Image", type="pil")
            search_queries = gr.Textbox(
                label="Search Queries (comma-separated)",
                placeholder="cat, dog, car, tree, house"
            )
            search_btn = gr.Button("Search", variant="primary")
            
        with gr.Column():
            search_rankings = gr.Textbox(
                label="Query Rankings",
                interactive=False,
                lines=8
            )
            search_details = gr.Textbox(
                label="Details",
                interactive=False,
                lines=4
            )
    
    search_btn.click(
        fn=text_search,
        inputs=[search_image, search_queries],
        outputs=[search_rankings, search_details]
    )
    
    return search_image, search_queries, search_rankings, search_details


def create_demo():
    """Create the main Gradio demo interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    .gr-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    """
    
    with gr.Blocks(css=css, title="OmniVision Demo") as demo:
        
        # Header
        gr.Markdown("""
        # 🎯 OmniVision: Text-Guided Segmentation
        
        **Describe any object in natural language and watch it get segmented automatically!**
        
        Powered by **Trident-Inspired Feature Splicing** combining DINOv3, SAM 2, and CLIP
        """)
        
        if not MODELS_AVAILABLE:
            gr.Markdown("⚠️ **Models not available.** Please install required dependencies.")
            return demo
        
        # Simple Text Segmentation Interface
        gr.Markdown("""
        ### 🗣️ Text-Guided Object Segmentation
        Upload an image and describe the object you want to segment in natural language.
        
        **✨ Now with Trident-Inspired Feature Splicing for much better results!**
        """)
        create_text_interface()
        
        # Add some helpful tips
        gr.Markdown("""
        ### 💡 Tips for better results:
        - Use simple, direct descriptions: "car", "person", "bottle"
        - Try color + object: "red car", "blue shirt"  
        - **New with Trident**: Higher thresholds (0.8-0.9) work well now!
        - Lower threshold (0.7-0.8) finds more matches
        - Higher threshold (0.85-0.95) gives precise matches
        - First run takes 1-2 minutes to load models
        """)
        
        # Footer
        gr.Markdown("""
        ---
        **🤖 Powered by:** DINOv3 (Meta) + SAM 2 (Meta) + CLIP (OpenAI)
        
        **📚 Learn more:** [DINOv3](https://github.com/facebookresearch/dinov3) | [SAM 2](https://github.com/facebookresearch/segment-anything-2) | [CLIP](https://github.com/openai/CLIP)
        """)
    
    return demo


def launch_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """
    Launch the Gradio demo
    
    Args:
        server_name: Server hostname
        server_port: Server port
        share: Whether to create a public link
        debug: Enable debug mode
    """
    if not MODELS_AVAILABLE:
        print("❌ Cannot launch demo: Models not available")
        print("Please install required dependencies:")
        print("  pip install openai-clip")
        print("  conda install conda-forge::sam-2")
        return
    
    demo = create_demo()
    
    print(f"🚀 Launching OmniVision Demo...")
    print(f"📍 Server: http://{server_name}:{server_port}")
    
    if share:
        print("🌐 Creating public link...")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=True
    )


if __name__ == "__main__":
    launch_demo(debug=True)