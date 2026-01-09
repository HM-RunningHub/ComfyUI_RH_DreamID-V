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
import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import multiprocessing
import glob

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

import insightface
from insightface.app import FaceAnalysis

CUR_DIR = os.path.dirname(__file__)
import folder_paths

class LMKExtractor():
    def __init__(self, FPS=25):
        # Create an FaceLandmarker object.
        self.mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
        base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/face_landmarker_v2_with_blendshapes.task'))
        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=self.mode,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = face_landmark.FaceLandmarker.create_from_options(options)
        self.last_ts = 0
        self.frame_ms = int(1000 / FPS)

        det_base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models/blaze_face_short_range.tflite'))
        det_options = vision.FaceDetectorOptions(base_options=det_base_options)
        self.det_detector = vision.FaceDetector.create_from_options(det_options)

        #kiki
        self.handler = FaceAnalysis(allowed_modules=['detection'], 
            providers=['CPUExecutionProvider'],
            root=os.path.join(folder_paths.models_dir, 'insightface')
            )
        self.handler.prepare(ctx_id=0, det_size=(640, 640))
                

    def __call__(self, img):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        t0 = time.time()

        do_insightface = False

        if self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.VIDEO:
            det_result = self.det_detector.detect(image)
            if len(det_result.detections) != 1:
                return None
            self.last_ts += self.frame_ms
            try:
                detection_result, mesh3d = self.detector.detect_for_video(image, timestamp_ms=self.last_ts)
            except:
                return None
        elif self.mode == mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE:
            # det_result = self.det_detector.detect(image)

            # if len(det_result.detections) != 1:
            #     return None
            try:
                detection_result, mesh3d = self.detector.detect(image)
            except:
                faces = self.handler.get(frame)
                if len(faces) < 1:
                    return None
                h_orig, w_orig = frame.shape[:2]
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                x1, y1, x2, y2 = face.bbox.astype(int)
                box_w = x2 - x1
                box_h = y2 - y1
                padding = 0.4 
                
                cx1 = max(0, int(x1 - box_w * padding))
                cy1 = max(0, int(y1 - box_h * padding))
                cx2 = min(w_orig, int(x2 + box_w * padding))
                cy2 = min(h_orig, int(y2 + box_h * padding))
                
                roi_img = frame[cy1:cy2, cx1:cx2]
                roi_h, roi_w = roi_img.shape[:2]
                roi_img_contiguous = np.ascontiguousarray(roi_img)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_img_contiguous)
                try:
                    do_insightface = True
                    detection_result, mesh3d = self.detector.detect(mp_image)
                    # print('face detection success')
                    for landmark in detection_result.face_landmarks[0]:
                        landmark.x = (landmark.x * roi_w + cx1) / w_orig
                        landmark.y = (landmark.y * roi_h + cy1) / h_orig
                    # rescaledlmks3d = np.array(mesh3d.vertex_buffer)
                    # rescaledlmks3d[:, 0] = (rescaledlmks3d[:, 0] * roi_w + cx1) / w_orig
                    # rescaledlmks3d[:, 1] = (rescaledlmks3d[:, 1] * roi_h + cy1) / h_orig
                except:
                    print('still can not detect face')
                    return None
                
                # return None
            
        
        bs_list = detection_result.face_blendshapes
        if len(bs_list) == 1:
            bs = bs_list[0]
            bs_values = []
            for index in range(len(bs)):
                bs_values.append(bs[index].score)
            bs_values = bs_values[1:] # remove neutral
            trans_mat = detection_result.facial_transformation_matrixes[0]
            face_landmarks_list = detection_result.face_landmarks
            face_landmarks = face_landmarks_list[0]
            lmks = []
            for index in range(len(face_landmarks)):
                x = face_landmarks[index].x
                y = face_landmarks[index].y
                z = face_landmarks[index].z
                lmks.append([x, y, z])
            lmks = np.array(lmks)
            
            # if do_insightface:
            #     lmks3d = rescaledlmks3d
            # else:   
            lmks3d = np.array(mesh3d.vertex_buffer)
            lmks3d = lmks3d.reshape(-1, 5)[:, :3]
            if do_insightface:
                lmks3d[:, 0] = (lmks3d[:, 0] * roi_w + cx1) / w_orig
                lmks3d[:, 1] = (lmks3d[:, 1] * roi_h + cy1) / h_orig
            mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1

            return {
                "lmks": lmks,
                'lmks3d': lmks3d,
                "trans_mat": trans_mat,
                'faces': mp_tris,
                "bs": bs_values
            }
        else:
            print('multiple faces in the image')
            return None
        