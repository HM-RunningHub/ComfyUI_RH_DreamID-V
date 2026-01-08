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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

CUR_DIR = os.path.dirname(__file__)


class InsightFaceDetector:
    """InsightFace-based face detector for robust detection."""
    
    def __init__(self, det_size=(640, 640), det_thresh=0.3):
        """
        Initialize InsightFace detector.
        
        Args:
            det_size: Detection input size
            det_thresh: Detection threshold (lower = more detections)
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not installed. Run: pip install insightface onnxruntime")
        
        self.app = FaceAnalysis(
            allowed_modules=['detection'],  # Only use detection module
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self.det_thresh = det_thresh
        print(f"[InsightFaceDetector] Initialized with det_thresh={det_thresh}")
    
    def detect(self, img_bgr):
        """
        Detect faces in image.
        
        Args:
            img_bgr: BGR image (OpenCV format)
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        faces = self.app.get(img_bgr)
        bboxes = []
        for face in faces:
            bbox = face.bbox.astype(int)
            bboxes.append(bbox)
        return bboxes


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
            bboxes = self.insightface_detector.detect(img_bgr)
            
            if len(bboxes) == 0:
                if debug:
                    print(f"[HybridLMKExtractor] InsightFace: no face detected")
                # Fall back to MediaPipe direct detection
                return self._extract_landmarks_mediapipe(img_bgr, debug)
            
            if len(bboxes) > 1:
                if debug:
                    print(f"[HybridLMKExtractor] InsightFace: {len(bboxes)} faces, using largest")
                # Use the largest face
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in bboxes]
                bboxes = [bboxes[np.argmax(areas)]]
            
            # Crop face region and run MediaPipe
            bbox = bboxes[0]
            cropped, offset, crop_size = self._crop_and_pad_face(img_bgr, bbox)
            
            if debug:
                print(f"[HybridLMKExtractor] InsightFace detected face at {bbox}, cropped to {crop_size}")
            
            # Run MediaPipe on cropped image
            result = self._extract_landmarks_mediapipe(cropped, debug)
            
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
            else:
                if debug:
                    print(f"[HybridLMKExtractor] MediaPipe failed on cropped face, trying full image")
                # Fall back to full image
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

