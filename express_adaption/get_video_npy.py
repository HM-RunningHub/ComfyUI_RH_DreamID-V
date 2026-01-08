# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import os
import cv2
import csv
import json
import math
import random
import numpy as np
import imageio
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageSequence
from io import BytesIO
from IPython.display import Video
from IPython.display import display, Image as IPyImage
import torchvision.transforms as T

import sys
from .media_pipe.mp_utils import LMKExtractor
from .media_pipe.draw_util import FaceMeshVisualizer
from .media_pipe.pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix

# Try to import hybrid detector (InsightFace + MediaPipe)
HYBRID_AVAILABLE = False
try:
    from .media_pipe.face_detector_hybrid import HybridLMKExtractor, INSIGHTFACE_AVAILABLE
    HYBRID_AVAILABLE = INSIGHTFACE_AVAILABLE
    if HYBRID_AVAILABLE:
        print("[get_video_npy] InsightFace hybrid detector available")
except ImportError as e:
    print(f"[get_video_npy] Hybrid detector not available: {e}")

# Default global extractor for backward compatibility (used by get_video_npy)
# Uses default threshold of 0.5
lmk_extractor = LMKExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
vis = FaceMeshVisualizer(forehead_edge=False)

def prehandle_video(video_path, save_path, fps=24, debug=False, min_detection_confidence=0.5, use_insightface=False):
    """
    Preprocess video: filter frames with detectable faces and save face detection results.
    
    Args:
        video_path: Path to input video
        save_path: Path to save filtered video
        fps: Frames per second
        debug: Enable debug logging
        min_detection_confidence: Face detection threshold (0.1-1.0, lower = more detections)
        use_insightface: Use InsightFace + MediaPipe hybrid detection (better for difficult videos)
    
    Returns:
        skip_frames_index: list of frame indices that were skipped (no face detected)
        skip_frames_data: dict mapping frame index to frame data
        face_results: list of face detection results for frames with faces
    """
    # Create extractor with custom threshold
    if use_insightface and HYBRID_AVAILABLE:
        from .media_pipe.face_detector_hybrid import HybridLMKExtractor
        extractor = HybridLMKExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            use_insightface=True,
            insightface_det_thresh=min_detection_confidence
        )
        print(f"[prehandle_video] Using InsightFace + MediaPipe hybrid detector (threshold: {min_detection_confidence})")
    else:
        extractor = LMKExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        print(f"[prehandle_video] Using MediaPipe detector (threshold: {min_detection_confidence})")
    
    frames = imageio.get_reader(video_path)
    meta = frames.get_meta_data()

    # size = meta.get('size')
    codec = meta.get('codec', 'libx264')
    writer = imageio.get_writer(
        save_path, 
        fps=fps, 
        codec=codec, 
        macro_block_size=1,
        quality=10
    )
    skip_frames_index = []
    skip_frames_data = {}
    face_results = []  # Store face results to avoid re-detection
    total_frames = 0
    
    # Only enable debug for first few frames to avoid log spam
    debug_limit = 10 if debug else 0
    
    for i, frame in enumerate(frames):
        total_frames += 1
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Enable debug for first N frames
        enable_debug = debug and i < debug_limit
        face_result = extractor(frame_bgr, debug=enable_debug)
        
        if face_result is None:
            skip_frames_index.append(i)
            skip_frames_data[i] = frame
            continue
        
        # Save face result with frame dimensions
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        face_results.append(face_result)
        
        writer.append_data(frame)
    writer.close()
    
    detected_count = total_frames - len(skip_frames_index)
    print(f"[prehandle_video] Total frames: {total_frames}, detected faces: {detected_count}, skipped: {len(skip_frames_index)}")
    
    return skip_frames_index, skip_frames_data, face_results

def get_video_npy(video_path):
    """
    Extract face landmarks from video frames.
    Frames without detectable faces are skipped.
    """
    frames = imageio.get_reader(video_path)

    face_results = []
    skipped_frames = 0
    total_frames = 0
    
    for i, frame in enumerate(frames):
        total_frames += 1
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        face_result = lmk_extractor(frame_bgr)
        
        # Skip frames without detected face
        if face_result is None:
            skipped_frames += 1
            continue
            
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        
        face_results.append(face_result)
    
    # Log skipped frames info
    if skipped_frames > 0:
        print(f"[get_video_npy] Skipped {skipped_frames}/{total_frames} frames without detected face.")
    
    # Ensure at least some frames have faces
    assert len(face_results) > 0, "Can not detect a face in any frame of the reference video."
    
    return face_results
    


if __name__ == '__main__':
    video_path = '/mnt/bn/hmbytenas/code/guoxu/guoxu/video_face_swap_test/videos/male_005.mp4'
    npy_root_path = '/mnt/bn/hmbytenas/code/guoxu/guoxu/video_face_swap_test/video_npy'
    face_results = get_video_npy(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    save_path = os.path.join(npy_root_path, f'{video_name}.npy')
    np.save(save_path, face_results)
    print(save_path, 'done')
