"""
Command Line Interface for OmniVision

Provides commands for testing models, computing similarities, and performing localization.
"""

import click
import torch
import time
from pathlib import Path
from PIL import Image
import numpy as np
import json

from ..models.dinov3_backbone import DINOv3Backbone, create_backbone
from ..pipelines.similarity import SimilarityPipeline
from ..pipelines.localization import LocalizationPipeline

try:
    from ..pipelines.segmentation import SegmentationPipeline
    from ..pipelines.video import VideoPipeline
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

try:
    from ..pipelines.text_guided import TextGuidedPipeline
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """OmniVision: Vision Bridge for Self-Supervised Segmentation"""
    pass


@cli.command()
@click.option('--model', '-m', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='Model name or short name (small, base, large, 7b)')
@click.option('--device', '-d', default=None,
              help='Device to use (cpu, mps, auto)')
@click.option('--image', '-i', type=click.Path(exists=True),
              help='Test image path (optional)')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def test_model(model, device, image, verbose):
    """Test DINOv3 model loading and basic functionality"""
    
    click.echo(f"Testing DINOv3 model: {model}")
    
    if verbose:
        import psutil
        click.echo(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        click.echo(f"PyTorch version: {torch.__version__}")
        click.echo(f"MPS available: {torch.backends.mps.is_available()}")
    
    try:
        # Load model
        start_time = time.time()
        backbone = DINOv3Backbone(model_name=model, device=device)
        load_time = time.time() - start_time
        
        click.echo(f"✅ Model loaded in {load_time:.2f}s")
        
        if verbose:
            info = backbone.get_model_info()
            click.echo(f"Parameters: {info['num_parameters']/1e6:.1f}M")
            click.echo(f"Hidden size: {info['hidden_size']}")
            click.echo(f"Device: {info['device']}")
            click.echo(f"Dtype: {info['dtype']}")
        
        # Test feature extraction
        if image:
            click.echo(f"Testing feature extraction on: {image}")
            start_time = time.time()
            features = backbone.extract_features(image)
            inference_time = time.time() - start_time
            
            click.echo(f"✅ Feature extraction completed in {inference_time:.3f}s")
            click.echo(f"CLS shape: {features['cls'].shape}")
            click.echo(f"Patch grid shape: {features['patch_grid'].shape}")
        else:
            # Test with dummy image
            click.echo("Testing with dummy image...")
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            start_time = time.time()
            features = backbone.extract_features(dummy_img)
            inference_time = time.time() - start_time
            
            click.echo(f"✅ Dummy inference completed in {inference_time:.3f}s")
            click.echo(f"Output shapes: CLS {features['cls'].shape}, Patches {features['patch_grid'].shape}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())


@cli.command()
@click.option('--img1', '-1', required=True, type=click.Path(exists=True),
              help='First image path')
@click.option('--img2', '-2', required=True, type=click.Path(exists=True),
              help='Second image path')
@click.option('--model', '-m', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='Model to use')
@click.option('--method', default="cls", type=click.Choice(['cls', 'patches', 'mixed']),
              help='Similarity computation method')
@click.option('--correspondences', '-c', type=int, default=10,
              help='Number of correspondences to find')
@click.option('--output', '-o', type=click.Path(),
              help='Output JSON file for results')
@click.option('--visualize', is_flag=True,
              help='Create visualization (requires matplotlib)')
def similarity(img1, img2, model, method, correspondences, output, visualize):
    """Compute similarity between two images"""
    
    click.echo(f"Computing similarity between {img1} and {img2}")
    
    try:
        # Initialize pipeline
        pipeline = SimilarityPipeline(model_name=model)
        
        # Overall image similarity
        start_time = time.time()
        overall_sim = pipeline.compute_image_similarity(img1, img2, method=method)
        sim_time = time.time() - start_time
        
        click.echo(f"Overall similarity ({method}): {overall_sim:.3f}")
        click.echo(f"Computed in {sim_time:.3f}s")
        
        # Find correspondences
        if correspondences > 0:
            click.echo(f"Finding {correspondences} correspondences...")
            start_time = time.time()
            matches = pipeline.find_correspondences(img1, img2, num_matches=correspondences)
            match_time = time.time() - start_time
            
            click.echo(f"Found {len(matches)} correspondences in {match_time:.3f}s")
            
            # Display top matches
            for i, match in enumerate(matches[:5]):
                click.echo(f"  {i+1}: {match['coord1']} -> {match['coord2']} (sim: {match['similarity']:.3f})")
            
            if len(matches) > 5:
                click.echo(f"  ... and {len(matches)-5} more")
        
        # Save results
        if output:
            results = {
                'img1': str(img1),
                'img2': str(img2),
                'model': model,
                'method': method,
                'overall_similarity': float(overall_sim),
                'correspondences': matches if correspondences > 0 else []
            }
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            
            click.echo(f"Results saved to {output}")
        
        # Visualization
        if visualize:
            try:
                import matplotlib.pyplot as plt
                from ..utils.visualization import visualize_correspondences
                
                if correspondences > 0:
                    fig = visualize_correspondences(img1, img2, matches)
                    output_path = f"similarity_viz_{Path(img1).stem}_{Path(img2).stem}.png"
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    click.echo(f"Visualization saved to {output_path}")
                    plt.close(fig)
                else:
                    click.echo("No correspondences to visualize")
                    
            except ImportError:
                click.echo("⚠️  Visualization requires matplotlib")
            except Exception as e:
                click.echo(f"⚠️  Visualization failed: {e}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--ref', '-r', required=True, type=click.Path(exists=True),
              help='Reference image path')
@click.option('--target', '-t', type=click.Path(exists=True),
              help='Target image path (if different from reference)')
@click.option('--click', 'click_coords_str', required=True,
              help='Click coordinates as "x,y"')
@click.option('--model', '-m', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='Model to use')
@click.option('--threshold', default=0.6, type=float,
              help='Similarity threshold')
@click.option('--region-size', default=1, type=int,
              help='Size of reference region (in patches)')
@click.option('--output', '-o', type=click.Path(),
              help='Output JSON file for results')
@click.option('--visualize', is_flag=True,
              help='Create visualization')
def localize(ref, target, click_coords_str, model, threshold, region_size, output, visualize):
    """Localize objects by clicking on reference image"""
    
    # Parse click coordinates
    try:
        x, y = map(int, click_coords_str.split(','))
        click_coords = (x, y)
    except ValueError:
        click.echo("❌ Invalid click coordinates. Use format: x,y")
        return
    
    target_path = target if target else ref
    
    click.echo(f"Localizing from click at {click_coords} in {ref}")
    click.echo(f"Searching in {target_path}")
    
    try:
        # Initialize pipeline
        pipeline = LocalizationPipeline(model_name=model)
        
        # Perform localization
        start_time = time.time()
        result = pipeline.localize_by_click(
            ref,
            click_coords,
            target_path,
            similarity_threshold=threshold,
            region_size=region_size
        )
        loc_time = time.time() - start_time
        
        click.echo(f"Localization completed in {loc_time:.3f}s")
        click.echo(f"Reference patch: {result['reference_patch']}")
        click.echo(f"Found {len(result['detections'])} detections")
        
        # Display top detections
        for i, det in enumerate(result['detections'][:5]):
            bbox = det['bbox']
            click.echo(f"  {i+1}: bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) sim={det['similarity']:.3f}")
        
        # Save results
        if output:
            # Convert tensors to lists for JSON serialization
            output_result = {
                'reference_image': str(ref),
                'target_image': str(target_path),
                'click_coords': click_coords,
                'reference_patch': result['reference_patch'],
                'model': model,
                'threshold': threshold,
                'region_size': region_size,
                'detections': result['detections'],
                'similarity_map_shape': list(result['similarity_map'].shape)
            }
            
            with open(output, 'w') as f:
                json.dump(output_result, f, indent=2)
            
            click.echo(f"Results saved to {output}")
        
        # Visualization
        if visualize:
            try:
                import matplotlib.pyplot as plt
                from ..utils.visualization import visualize_localization
                
                fig = visualize_localization(target_path, result)
                output_path = f"localization_viz_{Path(ref).stem}_{Path(target_path).stem}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                click.echo(f"Visualization saved to {output_path}")
                plt.close(fig)
                
            except ImportError:
                click.echo("⚠️  Visualization requires matplotlib")
            except Exception as e:
                click.echo(f"⚠️  Visualization failed: {e}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Image to extract features from')
@click.option('--model', '-m', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='Model to use')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for features (.npz format)')
@click.option('--token-type', default="all", type=click.Choice(['cls', 'patches', 'registers', 'all']),
              help='Type of tokens to extract')
def extract_features(image, model, output, token_type):
    """Extract DINOv3 features from an image"""
    
    click.echo(f"Extracting {token_type} features from {image}")
    
    try:
        # Initialize backbone
        backbone = DINOv3Backbone(model_name=model)
        
        # Extract features
        start_time = time.time()
        features = backbone.extract_features(image)
        extract_time = time.time() - start_time
        
        click.echo(f"Feature extraction completed in {extract_time:.3f}s")
        
        # Display shapes
        if token_type in ['cls', 'all']:
            click.echo(f"CLS token shape: {features['cls'].shape}")
        if token_type in ['patches', 'all']:
            click.echo(f"Patch tokens shape: {features['patches'].shape}")
            click.echo(f"Patch grid shape: {features['patch_grid'].shape}")
        if token_type in ['registers', 'all']:
            click.echo(f"Register tokens shape: {features['registers'].shape}")
        
        # Save features
        if output:
            save_dict = {}
            
            if token_type in ['cls', 'all']:
                save_dict['cls'] = features['cls'].cpu().numpy()
            if token_type in ['patches', 'all']:
                save_dict['patches'] = features['patches'].cpu().numpy()
                save_dict['patch_grid'] = features['patch_grid'].cpu().numpy()
            if token_type in ['registers', 'all']:
                save_dict['registers'] = features['registers'].cpu().numpy()
            
            # Add metadata
            save_dict['model'] = model
            save_dict['image_path'] = str(image)
            
            np.savez(output, **save_dict)
            click.echo(f"Features saved to {output}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--model', '-m', default="small",
              help='Model size to benchmark (small, base, large, 7b)')
@click.option('--num-images', '-n', default=10, type=int,
              help='Number of test images')
@click.option('--image-size', default=224, type=int,
              help='Test image size')
def benchmark(model, num_images, image_size):
    """Benchmark model performance"""
    
    click.echo(f"Benchmarking {model} model with {num_images} images of size {image_size}x{image_size}")
    
    try:
        # Create backbone
        backbone = create_backbone(model)
        
        # Generate test images
        test_images = []
        for i in range(num_images):
            img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            test_images.append(Image.fromarray(img_array))
        
        # Warmup
        click.echo("Warming up...")
        _ = backbone.extract_features(test_images[0])
        
        # Benchmark
        click.echo("Running benchmark...")
        start_time = time.time()
        
        for img in test_images:
            features = backbone.extract_features(img)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_images
        
        click.echo(f"Total time: {total_time:.3f}s")
        click.echo(f"Average time per image: {avg_time:.3f}s")
        click.echo(f"Throughput: {1/avg_time:.1f} images/second")
        
        # Memory info
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            click.echo(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            pass
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Image to segment')
@click.option('--click', 'click_coords_str',
              help='Click coordinates as "x,y" for interactive segmentation')
@click.option('--ref-image', type=click.Path(exists=True),
              help='Reference image for patch-based segmentation')
@click.option('--bbox', 
              help='Reference bounding box as "x,y,w,h" (requires --ref-image)')
@click.option('--dinov3-model', '-d', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='DINOv3 model to use')
@click.option('--sam2-model', '-s', default="tiny", 
              type=click.Choice(['tiny', 'small', 'base', 'large']),
              help='SAM 2 model size')
@click.option('--threshold', '-t', default=0.6, type=float,
              help='Similarity threshold')
@click.option('--no-sam2', is_flag=True,
              help='Skip SAM 2 refinement (DINOv3 similarity only)')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--visualize', is_flag=True,
              help='Create visualization')
def segment(image, click_coords_str, ref_image, bbox, dinov3_model, sam2_model, 
           threshold, no_sam2, output, visualize):
    """Segment objects using DINOv3 + SAM 2"""
    
    if not SAM2_AVAILABLE:
        click.echo("❌ SAM 2 not available. Install with: conda install conda-forge::sam-2")
        return
    
    click.echo(f"Segmenting image: {image}")
    
    try:
        # Initialize pipeline
        pipeline = SegmentationPipeline(
            dinov3_model=dinov3_model,
            sam2_model=sam2_model
        )
        
        # Determine segmentation mode
        if click_coords_str:
            # Click-based segmentation
            try:
                x, y = map(int, click_coords_str.split(','))
                click_coords = (x, y)
            except ValueError:
                click.echo("❌ Invalid click coordinates. Use format: x,y")
                return
            
            click.echo(f"Click-based segmentation at {click_coords}")
            result = pipeline.segment_by_click(
                image,
                click_coords,
                similarity_threshold=threshold,
                sam2_refine=not no_sam2
            )
            
        elif bbox and ref_image:
            # Patch-based segmentation
            try:
                x, y, w, h = map(int, bbox.split(','))
                bbox_coords = (x, y, w, h)
            except ValueError:
                click.echo("❌ Invalid bbox coordinates. Use format: x,y,w,h")
                return
            
            click.echo(f"Patch-based segmentation using bbox {bbox_coords}")
            result = pipeline.segment_by_patch(
                image,
                bbox_coords,
                ref_image,
                similarity_threshold=threshold,
                sam2_refine=not no_sam2
            )
            
        else:
            click.echo("❌ Must specify either --click or --bbox with --ref-image")
            return
        
        # Display results
        masks = result.get('masks', [])
        if isinstance(masks, np.ndarray) and masks.size > 0:
            num_masks = len(masks) if masks.ndim == 3 else 1
        else:
            num_masks = len(masks) if isinstance(masks, list) else 0
            
        click.echo(f"✅ Segmentation complete!")
        click.echo(f"Found {num_masks} mask(s)")
        
        if 'similarity_map' in result:
            sim_shape = result['similarity_map'].shape
            click.echo(f"Similarity map: {sim_shape}")
        
        # Save results
        if output:
            import os
            os.makedirs(output, exist_ok=True)
            
            # Save masks
            for i, mask in enumerate(masks):
                if isinstance(mask, np.ndarray) and mask.size > 0:
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_path = os.path.join(output, f"mask_{i:03d}.png")
                    mask_img.save(mask_path)
                    click.echo(f"Saved mask {i} to {mask_path}")
            
            # Save similarity map
            if 'similarity_map' in result:
                sim_map = result['similarity_map']
                sim_normalized = ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) * 255).astype(np.uint8)
                sim_img = Image.fromarray(sim_normalized)
                sim_path = os.path.join(output, "similarity_map.png")
                sim_img.save(sim_path)
                click.echo(f"Saved similarity map to {sim_path}")
        
        # Visualization
        if visualize:
            try:
                import matplotlib.pyplot as plt
                
                # Create visualization
                vis_img = pipeline.visualize_segmentation(
                    image,
                    result,
                    show_similarity=True,
                    show_masks=True
                )
                
                # Display
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(vis_img)
                ax.set_title('DINOv3 + SAM 2 Segmentation')
                ax.axis('off')
                
                # Add click point if available
                if click_coords_str:
                    ax.plot(click_coords[0], click_coords[1], 'r*', markersize=15, label='Click')
                    ax.legend()
                
                output_path = f"segmentation_{Path(image).stem}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                click.echo(f"Visualization saved to {output_path}")
                plt.close(fig)
                
            except ImportError:
                click.echo("⚠️  Visualization requires matplotlib")
            except Exception as e:
                click.echo(f"⚠️  Visualization failed: {e}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True),
              help='Input video file')
@click.option('--frame', '-f', type=int, default=0,
              help='Reference frame index for initial prompt')
@click.option('--click', 'click_coords_str',
              help='Click coordinates as "x,y" for initial prompt')
@click.option('--bbox', 
              help='Initial bounding box as "x,y,w,h"')
@click.option('--dinov3-model', '-d', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='DINOv3 model to use')
@click.option('--sam2-model', '-s', default="tiny", 
              type=click.Choice(['tiny', 'small', 'base', 'large']),
              help='SAM 2 model size')
@click.option('--max-frames', '-m', type=int,
              help='Maximum number of frames to process')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--similarity-only', is_flag=True,
              help='Use only DINOv3 similarity tracking (no SAM 2)')
@click.option('--threshold', '-t', default=0.6, type=float,
              help='Similarity threshold for tracking')
@click.option('--temporal-smoothing', is_flag=True, default=True,
              help='Apply temporal smoothing to tracking')
def track(video, frame, click_coords_str, bbox, dinov3_model, sam2_model, 
          max_frames, output, similarity_only, threshold, temporal_smoothing):
    """Track objects in video using DINOv3 + SAM 2"""
    
    if not SAM2_AVAILABLE:
        click.echo("❌ SAM 2 not available. Install with: conda install conda-forge::sam-2")
        return
    
    if not click_coords_str and not bbox:
        click.echo("❌ Must specify either --click or --bbox for initial prompt")
        return
    
    click.echo(f"Processing video: {video}")
    click.echo(f"Reference frame: {frame}")
    
    try:
        # Initialize pipeline
        pipeline = VideoPipeline(
            dinov3_model=dinov3_model,
            sam2_model=sam2_model
        )
        
        # Load video frames
        frames = pipeline.load_video_frames(video, max_frames)
        
        if frame >= len(frames):
            click.echo(f"❌ Reference frame {frame} >= total frames {len(frames)}")
            return
        
        if similarity_only:
            # DINOv3-only tracking
            if not click_coords_str:
                click.echo("❌ Click coordinates required for similarity-only tracking")
                return
            
            try:
                x, y = map(int, click_coords_str.split(','))
                reference_coords = (x, y)
            except ValueError:
                click.echo("❌ Invalid click coordinates. Use format: x,y")
                return
            
            click.echo(f"DINOv3 similarity tracking from click {reference_coords}")
            result = pipeline.track_by_similarity(
                frames,
                frame,
                reference_coords,
                similarity_threshold=threshold,
                temporal_smoothing=temporal_smoothing
            )
            
            # Display results
            click.echo(f"✅ Tracking complete!")
            click.echo(f"Processed {len(frames)} frames")
            
            # Show detection statistics
            total_detections = sum(len(dets) for dets in result['detections'])
            avg_detections = total_detections / len(frames)
            click.echo(f"Average detections per frame: {avg_detections:.2f}")
            
            if result['temporal_consistency']:
                avg_consistency = np.mean(result['temporal_consistency'])
                click.echo(f"Average temporal consistency: {avg_consistency:.3f}")
            
        else:
            # SAM 2 + DINOv3 tracking
            prompts = {}
            
            if click_coords_str:
                try:
                    x, y = map(int, click_coords_str.split(','))
                    prompts[frame] = {
                        'points': [(x, y)],
                        'labels': [1],
                        'object_id': 1
                    }
                except ValueError:
                    click.echo("❌ Invalid click coordinates. Use format: x,y")
                    return
            
            if bbox:
                try:
                    x, y, w, h = map(int, bbox.split(','))
                    if frame not in prompts:
                        prompts[frame] = {'object_id': 1}
                    prompts[frame]['boxes'] = [(x, y, x + w, y + h)]
                except ValueError:
                    click.echo("❌ Invalid bbox coordinates. Use format: x,y,w,h")
                    return
            
            click.echo(f"SAM 2 + DINOv3 tracking with prompts on frame {frame}")
            result = pipeline.track_with_sam2(
                frames,
                prompts,
                use_dinov3_guidance=True
            )
            
            # Display results
            click.echo(f"✅ Tracking complete!")
            click.echo(f"Processed {len(frames)} frames")
            click.echo(f"Tracked objects: {len(prompts)}")
            click.echo(f"Segmentation frames: {len(result['segments'])}")
        
        # Save results if output directory specified
        if output:
            import os
            import json
            os.makedirs(output, exist_ok=True)
            
            # Save tracking metadata
            metadata = {
                'video_path': str(video),
                'num_frames': len(frames),
                'reference_frame': frame,
                'dinov3_model': dinov3_model,
                'sam2_model': sam2_model,
                'similarity_only': similarity_only,
                'threshold': threshold
            }
            
            with open(os.path.join(output, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            click.echo(f"Results saved to {output}")
            
            # Save similarity maps if available
            if 'similarity_maps' in result:
                sim_dir = os.path.join(output, 'similarity_maps')
                os.makedirs(sim_dir, exist_ok=True)
                
                for i, sim_map in enumerate(result['similarity_maps']):
                    sim_normalized = ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) * 255).astype(np.uint8)
                    sim_img = Image.fromarray(sim_normalized)
                    sim_path = os.path.join(sim_dir, f"frame_{i:04d}.png")
                    sim_img.save(sim_path)
                
                click.echo(f"Saved {len(result['similarity_maps'])} similarity maps")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Image to segment')
@click.option('--text', '-t', required=True,
              help='Text description of object to segment')
@click.option('--dinov3-model', '-d', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='DINOv3 model to use')
@click.option('--clip-model', '-c', default="ViT-B/32",
              help='CLIP model to use')
@click.option('--sam2-model', '-s', default="tiny",
              type=click.Choice(['tiny', 'small', 'base', 'large']),
              help='SAM 2 model size')
@click.option('--threshold', '-th', default=0.3, type=float,
              help='Similarity threshold for segmentation')
@click.option('--text-weight', '-w', default=0.5, type=float,
              help='Weight for text similarity vs visual similarity')
@click.option('--no-sam2', is_flag=True,
              help='Skip SAM 2 refinement (similarity-only)')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--visualize', is_flag=True,
              help='Create visualization')
@click.option('--intermediate', is_flag=True,
              help='Return intermediate results (CLIP + DINOv3 separately)')
def text_segment(image, text, dinov3_model, clip_model, sam2_model, threshold, 
                text_weight, no_sam2, output, visualize, intermediate):
    """Segment objects using text descriptions"""
    
    if not CLIP_AVAILABLE:
        click.echo("❌ CLIP not available. Install with: pip install openai-clip")
        return
    
    if not SAM2_AVAILABLE and not no_sam2:
        click.echo("❌ SAM 2 not available. Install with: conda install conda-forge::sam-2")
        click.echo("    Use --no-sam2 for CLIP+DINOv3 only segmentation")
        return
    
    click.echo(f"Text-guided segmentation: '{text}'")
    click.echo(f"Image: {image}")
    
    try:
        # Initialize pipeline
        pipeline = TextGuidedPipeline(
            dinov3_model=dinov3_model,
            clip_model=clip_model,
            sam2_model=sam2_model,
            text_similarity_weight=text_weight
        )
        
        # Perform segmentation
        result = pipeline.segment_by_text(
            image,
            text,
            similarity_threshold=threshold,
            sam2_refine=not no_sam2,
            return_intermediate=intermediate
        )
        
        # Display results
        masks = result.get('masks', [])
        if isinstance(masks, np.ndarray) and masks.size > 0:
            num_masks = len(masks) if masks.ndim == 3 else 1
        else:
            num_masks = len(masks) if isinstance(masks, list) else 0
        
        click.echo(f"✅ Segmentation complete!")
        click.echo(f"Text query: '{text}'")
        click.echo(f"Found {num_masks} mask(s)")
        
        if 'combined_similarity' in result:
            sim_shape = result['combined_similarity'].shape
            sim_min = result['combined_similarity'].min()
            sim_max = result['combined_similarity'].max()
            click.echo(f"Similarity map: {sim_shape}, range: [{sim_min:.3f}, {sim_max:.3f}]")
        
        # Save results
        if output:
            import os
            import json
            os.makedirs(output, exist_ok=True)
            
            # Save masks
            for i, mask in enumerate(masks):
                if isinstance(mask, np.ndarray) and mask.size > 0:
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_path = os.path.join(output, f"text_mask_{i:03d}.png")
                    mask_img.save(mask_path)
                    click.echo(f"Saved mask {i} to {mask_path}")
            
            # Save similarity maps
            if 'combined_similarity' in result:
                sim_map = result['combined_similarity']
                sim_normalized = ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) * 255).astype(np.uint8)
                sim_img = Image.fromarray(sim_normalized)
                sim_path = os.path.join(output, "text_similarity_map.png")
                sim_img.save(sim_path)
                click.echo(f"Saved similarity map to {sim_path}")
            
            # Save intermediate results if available
            if intermediate:
                if 'clip_similarity' in result:
                    clip_sim = result['clip_similarity']
                    clip_normalized = ((clip_sim - clip_sim.min()) / (clip_sim.max() - clip_sim.min()) * 255).astype(np.uint8)
                    clip_img = Image.fromarray(clip_normalized)
                    clip_path = os.path.join(output, "clip_similarity.png")
                    clip_img.save(clip_path)
                
                if 'dinov3_similarity' in result:
                    dinov3_sim = result['dinov3_similarity']
                    dinov3_normalized = ((dinov3_sim - dinov3_sim.min()) / (dinov3_sim.max() - dinov3_sim.min()) * 255).astype(np.uint8)
                    dinov3_img = Image.fromarray(dinov3_normalized)
                    dinov3_path = os.path.join(output, "dinov3_similarity.png")
                    dinov3_img.save(dinov3_path)
            
            # Save metadata
            metadata = {
                'text_query': text,
                'image_path': str(image),
                'dinov3_model': dinov3_model,
                'clip_model': clip_model,
                'sam2_model': sam2_model,
                'threshold': threshold,
                'text_weight': text_weight,
                'sam2_refinement': not no_sam2,
                'num_masks': num_masks
            }
            
            with open(os.path.join(output, 'text_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Visualization
        if visualize and num_masks > 0:
            try:
                import matplotlib.pyplot as plt
                
                # Create visualization
                # This would use the segmentation pipeline's visualization method
                # For now, just display basic info
                click.echo("Visualization would be created here")
                
            except ImportError:
                click.echo("⚠️  Visualization requires matplotlib")
            except Exception as e:
                click.echo(f"⚠️  Visualization failed: {e}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Image to search')
@click.option('--queries', '-q', required=True,
              help='Comma-separated text queries to search for')
@click.option('--top-k', '-k', default=3, type=int,
              help='Number of top results to segment')
@click.option('--dinov3-model', '-d', default="facebook/dinov3-vits16-pretrain-lvd1689m",
              help='DINOv3 model to use')
@click.option('--clip-model', '-c', default="ViT-B/32",
              help='CLIP model to use')
@click.option('--sam2-model', '-s', default="tiny",
              type=click.Choice(['tiny', 'small', 'base', 'large']),
              help='SAM 2 model size')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
def text_search(image, queries, top_k, dinov3_model, clip_model, sam2_model, output):
    """Search for multiple objects and segment the most relevant ones"""
    
    if not CLIP_AVAILABLE:
        click.echo("❌ CLIP not available. Install with: pip install openai-clip")
        return
    
    if not SAM2_AVAILABLE:
        click.echo("❌ SAM 2 not available. Install with: conda install conda-forge::sam-2")
        return
    
    # Parse queries
    query_list = [q.strip() for q in queries.split(',')]
    
    click.echo(f"Searching for {len(query_list)} queries in {image}")
    for i, query in enumerate(query_list):
        click.echo(f"  {i+1}: '{query}'")
    
    try:
        # Initialize pipeline
        pipeline = TextGuidedPipeline(
            dinov3_model=dinov3_model,
            clip_model=clip_model,
            sam2_model=sam2_model
        )
        
        # Perform search and segmentation
        result = pipeline.search_and_segment(
            image,
            query_list,
            top_k=top_k,
            segment_top=True
        )
        
        # Display results
        click.echo(f"✅ Search complete!")
        
        # Show rankings
        click.echo("\nQuery Rankings:")
        for i, query_info in enumerate(result['query_rankings']):
            score = query_info['overall_score']
            query = query_info['query']
            click.echo(f"  {i+1}: '{query}' (score: {score:.3f})")
        
        # Show segmentation results
        segmentations = result['segmentations']
        click.echo(f"\nSegmented top {len(segmentations)} queries:")
        for query, seg_result in segmentations.items():
            masks = seg_result.get('masks', [])
            num_masks = len(masks) if isinstance(masks, list) else (len(masks) if hasattr(masks, '__len__') else 0)
            click.echo(f"  '{query}': {num_masks} mask(s)")
        
        # Save results
        if output:
            import os
            import json
            os.makedirs(output, exist_ok=True)
            
            # Save individual query results
            for query, seg_result in segmentations.items():
                query_dir = os.path.join(output, query.replace(' ', '_').replace('/', '_'))
                os.makedirs(query_dir, exist_ok=True)
                
                masks = seg_result.get('masks', [])
                for i, mask in enumerate(masks):
                    if isinstance(mask, np.ndarray) and mask.size > 0:
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_path = os.path.join(query_dir, f"mask_{i:03d}.png")
                        mask_img.save(mask_path)
            
            # Save search metadata
            metadata = {
                'image_path': str(image),
                'queries': query_list,
                'top_k': top_k,
                'rankings': result['query_rankings'],
                'dinov3_model': dinov3_model,
                'clip_model': clip_model,
                'sam2_model': sam2_model
            }
            
            with open(os.path.join(output, 'search_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            click.echo(f"Results saved to {output}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    cli()