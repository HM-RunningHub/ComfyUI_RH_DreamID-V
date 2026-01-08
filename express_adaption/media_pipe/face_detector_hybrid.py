# Copyright 2024-2025 RunningHub. All rights reserved.
# Hybrid face detector: InsightFace detection + MediaPipe landmarks
"""
Hybrid face detector that uses InsightFace for robust face detection
and MediaPipe for detailed landmark extraction.

This provides better face detection on challenging videos while
maintaining compatibility with the existing landmark-based pipeline.
"""

import os
import numpy as np
import cv2
import time

# Check if insightface is available
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("[HybridDetector] InsightFace not available, falling back to MediaPipe only")

# Get ComfyUI models path for InsightFace models
def get_insightface_model_root():
    """
    Get InsightFace model root directory.
    Priority:
    1. {ComfyUI}/models/insightface
    2. Default ~/.insightface
    """
    # Try to find ComfyUI models path
    try:
        import folder_paths
        comfy_models_path = folder_paths.models_dir
        insightface_path = os.path.join(comfy_models_path, "insightface")
        if os.path.exists(insightface_path):
            print(f"[HybridDetector] Using InsightFace models from: {insightface_path}")
            return insightface_path
    except ImportError:
        pass
    
    # Try relative path from this file (for standalone testing)
    # Go up to ComfyUI_RH_DreamID-V, then to parent ComfyUI/models/insightface
    current_dir = os.path.dirname(__file__)
    possible_paths = [
        # Standard ComfyUI custom_nodes layout
        os.path.join(current_dir, "..", "..", "..", "..", "models", "insightface"),
        # Alternative layout
        os.path.join(current_dir, "..", "..", "..", "models", "insightface"),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"[HybridDetector] Using InsightFace models from: {abs_path}")
            return abs_path
    
    # Fallback to default (will try to download if not exists)
    default_path = os.path.expanduser("~/.insightface")
    print(f"[HybridDetector] Using default InsightFace path: {default_path}")
    return default_path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

CUR_DIR = os.path.dirname(__file__)


class InsightFaceDetector:
    """InsightFace-based face detector for robust detection."""
    
    def __init__(self, det_size=(640, 640), det_thresh=0.3, model_name='buffalo_l'):
        """
        Initialize InsightFace detector.
        
        Args:
            det_size: Detection input size
            det_thresh: Detection threshold (lower = more detections)
            model_name: Model name (default: buffalo_l)
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not installed. Run: pip install insightface onnxruntime")
        
        # Get model root directory (use ComfyUI models path if available)
        model_root = get_insightface_model_root()
        
        # Check if model exists locally
        model_path = os.path.join(model_root, "models", model_name)
        if not os.path.exists(model_path):
            # Try without 'models' subdirectory
            model_path = os.path.join(model_root, model_name)
        
        if os.path.exists(model_path):
            print(f"[InsightFaceDetector] Found local model at: {model_path}")
        else:
            print(f"[InsightFaceDetector] WARNING: Model not found at {model_path}, may attempt download")
        
        self.app = FaceAnalysis(
            name=model_name,
            root=model_root,
            allowed_modules=['detection'],  # Only use detection module
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self.det_thresh = det_thresh
        print(f"[InsightFaceDetector] Initialized with det_thresh={det_thresh}, model={model_name}")
    
    def detect(self, img_bgr):
        """
        Detect faces in image.
        
        Args:
            img_bgr: BGR image (OpenCV format)
            
        Returns:
            List of tuples [(bbox, score, area), ...] where bbox is (x1, y1, x2, y2)
        """
        h, w = img_bgr.shape[:2]
        faces = self.app.get(img_bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Filter out invalid bboxes (negative coords or out of bounds)
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            
            # Filter out too small faces
            face_w, face_h = x2 - x1, y2 - y1
            if face_w < 30 or face_h < 30:
                continue
            
            area = face_w * face_h
            score = face.det_score if hasattr(face, 'det_score') else 1.0
            results.append((bbox, score, area))
        
        return results


class HybridLMKExtractor:
    """
    Hybrid face landmark extractor.
    Uses InsightFace for detection + MediaPipe for landmarks.
    Falls back to MediaPipe-only if InsightFace is not available.
    """
    
    def __init__(self, FPS=25, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 use_insightface=True, insightface_det_thresh=0.3):
        """
        Initialize hybrid extractor.
        
        Args:
            FPS: Frames per second for video mode
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
            use_insightface: Whether to use InsightFace for pre-detection
            insightface_det_thresh: InsightFace detection threshold
        """
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        
        # Initialize InsightFace detector
        if self.use_insightface:
            try:
                self.insightface_detector = InsightFaceDetector(det_thresh=insightface_det_thresh)
                print(f"[HybridLMKExtractor] Using InsightFace + MediaPipe hybrid mode")
            except Exception as e:
                print(f"[HybridLMKExtractor] Failed to init InsightFace: {e}, falling back to MediaPipe only")
                self.use_insightface = False
        
        # Initialize MediaPipe FaceLandmarker
        self.mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(CUR_DIR, 'mp_models/face_landmarker_v2_with_blendshapes.task')
        )
        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.mode,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_detector = face_landmark.FaceLandmarker.create_from_options(options)
        self.last_ts = 0
        self.frame_ms = int(1000 / FPS)
        
        if not self.use_insightface:
            print(f"[HybridLMKExtractor] Using MediaPipe only mode")
    
    def _crop_and_pad_face(self, img_bgr, bbox, padding_ratio=0.3):
        """
        Crop face region with padding.
        
        Args:
            img_bgr: Original image
            bbox: Face bounding box (x1, y1, x2, y2)
            padding_ratio: Padding ratio around face
            
        Returns:
            cropped_img: Cropped face image
            offset: (offset_x, offset_y) for coordinate mapping
            scale: Scale factor for coordinate mapping
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        face_w, face_h = x2 - x1, y2 - y1
        pad_w = int(face_w * padding_ratio)
        pad_h = int(face_h * padding_ratio)
        
        # Expand bbox with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop
        cropped = img_bgr[y1:y2, x1:x2].copy()
        
        return cropped, (x1, y1), (x2 - x1, y2 - y1)
    
    def _extract_landmarks_mediapipe(self, img_bgr, debug=False):
        """Extract landmarks using MediaPipe."""
        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        try:
            detection_result, mesh3d = self.mp_detector.detect(image)
        except Exception as e:
            if debug:
                print(f"[HybridLMKExtractor] MediaPipe detect exception: {e}")
            return None
        
        # Check if face was detected
        if mesh3d is None:
            return None
        
        bs_list = detection_result.face_blendshapes
        if len(bs_list) != 1:
            return None
        
        bs = bs_list[0]
        bs_values = [bs[i].score for i in range(len(bs))][1:]  # Remove neutral
        trans_mat = detection_result.facial_transformation_matrixes[0]
        face_landmarks = detection_result.face_landmarks[0]
        
        lmks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
        lmks3d = np.array(mesh3d.vertex_buffer).reshape(-1, 5)[:, :3]
        mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1
        
        return {
            "lmks": lmks,
            'lmks3d': lmks3d,
            "trans_mat": trans_mat,
            'faces': mp_tris,
            "bs": bs_values
        }
    
    def __call__(self, img_bgr, debug=False):
        """
        Extract face landmarks from image.
        
        Args:
            img_bgr: BGR image (OpenCV format)
            debug: Enable debug logging
            
        Returns:
            dict with lmks, lmks3d, trans_mat, faces, bs or None if no face
        """
        h, w = img_bgr.shape[:2]
        
        # Try InsightFace detection first
        if self.use_insightface:
            detections = self.insightface_detector.detect(img_bgr)
            
            if len(detections) == 0:
                if debug:
                    print(f"[HybridLMKExtractor] InsightFace: no valid face detected, trying MediaPipe")
                # Fall back to MediaPipe direct detection
                return self._extract_landmarks_mediapipe(img_bgr, debug)
            
            # Sort by score * area (prefer confident large faces)
            # Score weight is higher to prefer confident detections
            detections.sort(key=lambda x: x[1] * 2 + x[2] / (w * h), reverse=True)
            
            if debug and len(detections) > 1:
                print(f"[HybridLMKExtractor] InsightFace: {len(detections)} valid faces, using best")
            
            # Try faces in order of quality until one works with MediaPipe
            for bbox, score, area in detections[:3]:  # Try top 3 candidates
                cropped, offset, crop_size = self._crop_and_pad_face(img_bgr, bbox)
                
                if debug:
                    print(f"[HybridLMKExtractor] Trying face at {bbox.tolist()}, score={score:.2f}, size={crop_size}")
                
                # Run MediaPipe on cropped image
                result = self._extract_landmarks_mediapipe(cropped, debug=False)
                
                if result is not None:
                    # Map landmarks back to original image coordinates
                    crop_h, crop_w = cropped.shape[:2]
                    ox, oy = offset
                    
                    # lmks are normalized [0,1], need to map to original image
                    lmks = result['lmks'].copy()
                    lmks[:, 0] = (lmks[:, 0] * crop_w + ox) / w
                    lmks[:, 1] = (lmks[:, 1] * crop_h + oy) / h
                    result['lmks'] = lmks
                    
                    return result
            
            # All InsightFace candidates failed, try full image MediaPipe
            if debug:
                print(f"[HybridLMKExtractor] All InsightFace candidates failed, trying full image MediaPipe")
            return self._extract_landmarks_mediapipe(img_bgr, debug)
        else:
            # MediaPipe only mode
            return self._extract_landmarks_mediapipe(img_bgr, debug)


def get_extractor(use_insightface=True, min_detection_confidence=0.5, insightface_det_thresh=0.3):
    """
    Factory function to get the appropriate extractor.
    
    Args:
        use_insightface: Whether to try InsightFace + MediaPipe hybrid
        min_detection_confidence: MediaPipe detection confidence
        insightface_det_thresh: InsightFace detection threshold
        
    Returns:
        HybridLMKExtractor or LMKExtractor instance
    """
    if use_insightface and INSIGHTFACE_AVAILABLE:
        return HybridLMKExtractor(
            min_detection_confidence=min_detection_confidence,
            use_insightface=True,
            insightface_det_thresh=insightface_det_thresh
        )
    else:
        from .mp_utils import LMKExtractor
        return LMKExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

