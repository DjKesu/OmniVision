#!/usr/bin/env python3
"""
Create a simple test video for testing video tracking functionality
"""

import cv2
import numpy as np
from pathlib import Path

def create_moving_circle_video(output_path: str, num_frames: int = 30):
    """Create a video with a moving circle"""
    
    # Video properties
    width, height = 640, 480
    fps = 10
    
    # Circle properties
    radius = 30
    color = (0, 255, 0)  # Green circle
    thickness = -1  # Filled circle
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Frames: {num_frames}, Size: {width}x{height}, FPS: {fps}")
    
    for frame_idx in range(num_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate circle position (moving from left to right)
        progress = frame_idx / (num_frames - 1)
        center_x = int(radius + progress * (width - 2 * radius))
        center_y = height // 2
        
        # Add some vertical movement too
        center_y += int(50 * np.sin(progress * 4 * np.pi))
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), radius, color, thickness)
        
        # Add frame number text
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_idx % 10 == 0:
            print(f"  Written frame {frame_idx}/{num_frames}")
    
    # Release video writer
    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"Circle starts at (~{radius}, {height//2}) and moves to (~{width-radius}, {height//2})")

def create_multiple_objects_video(output_path: str, num_frames: int = 50):
    """Create a video with multiple moving objects"""
    
    # Video properties
    width, height = 640, 480
    fps = 10
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating multi-object test video: {output_path}")
    
    for frame_idx in range(num_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        progress = frame_idx / (num_frames - 1)
        
        # Object 1: Green circle (left to right)
        center1_x = int(50 + progress * (width - 100))
        center1_y = 150
        cv2.circle(frame, (center1_x, center1_y), 25, (0, 255, 0), -1)
        
        # Object 2: Red rectangle (right to left)
        center2_x = int(width - 50 - progress * (width - 100))
        center2_y = 350
        cv2.rectangle(frame, (center2_x - 20, center2_y - 20), 
                     (center2_x + 20, center2_y + 20), (0, 0, 255), -1)
        
        # Object 3: Blue triangle (circular motion)
        angle = progress * 4 * np.pi
        center3_x = int(width // 2 + 100 * np.cos(angle))
        center3_y = int(height // 2 + 80 * np.sin(angle))
        
        # Draw triangle
        pts = np.array([
            [center3_x, center3_y - 20],
            [center3_x - 17, center3_y + 10],
            [center3_x + 17, center3_y + 10]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (255, 0, 0))
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Multi-object test video created: {output_path}")

if __name__ == "__main__":
    # Create output directory
    Path("test_videos").mkdir(exist_ok=True)
    
    # Create test videos
    create_moving_circle_video("test_videos/moving_circle.mp4", num_frames=30)
    create_multiple_objects_video("test_videos/multiple_objects.mp4", num_frames=50)
    
    print("\n📹 Test videos created in test_videos/ directory")
    print("You can now test video tracking with:")
    print("  python -m dinov3_sam.cli.main track --video test_videos/moving_circle.mp4 --click 320,240 --max-frames 10")