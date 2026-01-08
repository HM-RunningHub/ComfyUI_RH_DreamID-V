#!/usr/bin/env python
"""
Standalone test script for face detection.
Run this directly without starting ComfyUI:
    python test_face_detection.py /path/to/your/video.mp4
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import imageio

def test_face_detection(video_path, max_frames=30, min_detection_confidence=0.3):
    """Test face detection on a video file."""
    
    # Import after path setup
    from express_adaption.media_pipe.mp_utils import LMKExtractor
    
    print(f"Testing face detection on: {video_path}")
    print(f"min_detection_confidence: {min_detection_confidence}")
    print("-" * 50)
    
    # Create extractor with custom threshold
    lmk_extractor = LMKExtractor(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence
    )
    
    # Read video
    frames = imageio.get_reader(video_path)
    
    detected = 0
    skipped = 0
    total = 0
    
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
            
        total += 1
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Enable debug for all frames
        face_result = lmk_extractor(frame_bgr, debug=True)
        
        if face_result is not None:
            detected += 1
            print(f"Frame {i}: ✓ Face detected")
        else:
            skipped += 1
            print(f"Frame {i}: ✗ No face")
    
    print("-" * 50)
    print(f"Results: {detected}/{total} frames with faces ({100*detected/total:.1f}%)")
    print(f"Skipped: {skipped} frames")
    
    return detected, total

def test_different_thresholds(video_path, max_frames=30):
    """Test with different detection thresholds."""
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    print("=" * 60)
    print("Testing different detection thresholds")
    print("=" * 60)
    
    results = []
    for thresh in thresholds:
        print(f"\n>>> Testing threshold: {thresh}")
        detected, total = test_face_detection(video_path, max_frames, thresh)
        results.append((thresh, detected, total))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for thresh, detected, total in results:
        print(f"  Threshold {thresh}: {detected}/{total} ({100*detected/total:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_face_detection.py <video_path> [max_frames]")
        print("Example: python test_face_detection.py /path/to/video.mp4 50")
        sys.exit(1)
    
    video_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Test with different thresholds to find the best one
    test_different_thresholds(video_path, max_frames)

