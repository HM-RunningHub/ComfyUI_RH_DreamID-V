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
from .media_pipe.mp_utils  import LMKExtractor
from .media_pipe.draw_util import FaceMeshVisualizer
from .media_pipe.pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix

lmk_extractor = LMKExtractor()
vis = FaceMeshVisualizer(forehead_edge=False)

def prehandle_video(video_path, save_path, fps=24):
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
    for i, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        face_result = lmk_extractor(frame_bgr)
        if face_result is None:
            print(f'frame {i} no face detected')
            skip_frames_index.append(i)
            skip_frames_data[i] = frame
            continue
        writer.append_data(frame)
        # print(f'frame {i} done')
    writer.close()
    return skip_frames_index, skip_frames_data

def get_video_npy(video_path):

    

    frames = imageio.get_reader(video_path)
    # print(f'frames count: {len(frames)}')

    face_results = []
    skip_frames_index = []
    skip_frames_data = {}
    detected_frames = []

    frame_width = 0
    frame_height = 0
    
    for i, frame in enumerate(frames):
        # if i == 0:
        #     frame_width = frame.shape[1]
        #     frame_height = frame.shape[0]
        #     print(f'frame size: {frame_width}x{frame_height}')
        #     while frame_width < 720 or frame_height < 720:
        #         frame_width *= 2
        #         frame_height *= 2
        #     print(f'frame resize to: {frame_width}x{frame_height}')
        # if frame_width != frame.shape[1]:
        #     frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

        # print(f'frame size: {frame.shape[1]}x{frame.shape[0]}')

        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        face_result = lmk_extractor(frame_bgr)
        # assert face_result is not None, "Can not detect a face in the reference image."
        if face_result is None:
            print(f'frame {i} no face detected')
            skip_frames_index.append(i)
            skip_frames_data[i] = frame
            continue
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        
        face_results.append(face_result)
        detected_frames.append(frame)
    return face_results, skip_frames_index, skip_frames_data, detected_frames
    


if __name__ == '__main__':
    video_path = '/mnt/bn/hmbytenas/code/guoxu/guoxu/video_face_swap_test/videos/male_005.mp4'
    npy_root_path = '/mnt/bn/hmbytenas/code/guoxu/guoxu/video_face_swap_test/video_npy'
    face_results = get_video_npy(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    save_path = os.path.join(npy_root_path, f'{video_name}.npy')
    np.save(save_path, face_results)
    print(save_path, 'done')
