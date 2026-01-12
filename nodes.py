import comfy.utils

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import uuid
import subprocess

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image, ImageOps


import cv2
import numpy as np
from .express_adaption.media_pipe import FaceMeshDetector, FaceMeshAlign_dreamidv
from .express_adaption.get_video_npy import get_video_npy
import folder_paths
from .express_adaption.get_video_npy import prehandle_video
try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None

def generate_pose_and_mask_videos(ref_video_path, ref_image_path, face_detection_threshold=0.5):

    print("Starting online generation of pose and mask videos...")
    detector = FaceMeshDetector()
    get_align_motion = FaceMeshAlign_dreamidv()
    CORE_LANDMARK_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
        300, 293, 334, 296, 336, 156,
    ]
    FACE_OVAL_INDICES = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    CORE_LANDMARK_INDICES.extend(FACE_OVAL_INDICES)
    CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES))

    def save_mask_or_points_frames(landmarks_sequence, frame_size, mode='points'):
        width, height = frame_size
        frames = []
        for frame_landmarks in landmarks_sequence:
            frame_image = np.zeros((height, width, 3), dtype=np.uint8)
            if mode == 'points':
                for landmark in frame_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
            elif mode == 'mask':
                face_oval_points = frame_landmarks.astype(np.int32)
                cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
            frames.append(frame_image)
        return frames

    def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps=30, mode='points'):
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_filename}")
            return
        print(f"Saving {mode} video to {output_filename}...")
        for frame_landmarks in landmarks_sequence:
            frame_image = np.zeros((height, width, 3), dtype=np.uint8)
            if mode == 'points':
                for landmark in frame_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
            elif mode == 'mask':
                face_oval_points = frame_landmarks.astype(np.int32)
                cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
            video_writer.write(frame_image)
        video_writer.release()
        print("Video saving complete.")
    fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
    # face_results = get_video_npy(ref_video_path)
    face_results, skip_frames_index, skip_frames_data, detected_frames = get_video_npy(ref_video_path, face_detection_threshold=face_detection_threshold)
    video_name = os.path.basename(ref_video_path).split('.')[0]
    #kiki:
    # temp_dir = os.path.join(os.path.dirname(ref_video_path), 'temp_generated')
    temp_dir = os.path.join(folder_paths.get_temp_directory(), 'dreamidv')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'try open ref_image_path:{ref_image_path}')
    image = Image.open(ref_image_path).convert('RGB')
    ref_image = np.array(image)
    _, ref_img_lmk = detector(ref_image)
    _, pose_addvis = get_align_motion(face_results, ref_img_lmk)
    width, height = face_results[0]['width'], face_results[0]['height']
 
    pose_output_path = os.path.join(temp_dir, video_name + '_pose.mp4')
    core_landmarks_sequence = pose_addvis[:, CORE_LANDMARK_INDICES, :]
    # save_visualization_video(
    #     landmarks_sequence=core_landmarks_sequence,
    #     output_filename=pose_output_path,
    #     frame_size=(width, height),
    #     fps=fps,
    #     mode='points'
    # )
    # mask_output_path = os.path.join(temp_dir, video_name + '_mask.mp4')
    # face_oval_sequence = pose_addvis[:, FACE_OVAL_INDICES, :]
    # save_visualization_video(
    #     landmarks_sequence=face_oval_sequence,
    #     output_filename=mask_output_path,
    #     frame_size=(width, height),
    #     fps=fps,
    #     mode='mask'
    # )
    # return pose_output_path, mask_output_path
    pose_frames = save_mask_or_points_frames(core_landmarks_sequence, (width, height), 'points')
    mask_frames = save_mask_or_points_frames(pose_addvis[:, FACE_OVAL_INDICES, :], (width, height), 'mask')
    return detected_frames, pose_frames, mask_frames, skip_frames_index, skip_frames_data

class RunningHub_DreamID_V_Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["origin", "faster"], {"default": "faster"}),
            }
        }

    RETURN_TYPES = ('RH_DreamID-V_Pipeline', )
    RETURN_NAMES = ('DreamID-V Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/DreamID-V"

    # OUTPUT_NODE = True

    def load(self, **kwargs):
        # hardcode
        task = 'swapface'
        ckpt_dir = os.path.join(folder_paths.models_dir, 'Wan', 'Wan2.1-T2V-1.3B')
        from .dreamidv_wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS[task]
        if kwargs.get('type') == 'faster':
            dreamidv_ckpt = os.path.join(folder_paths.models_dir, 'DreamID-V', 'dreamidv_faster.pth')
            print('use faster DreamID-V')
            from .dreamidv_wan_faster import DreamIDV as faster_DreamIDV
            wan_swapface = faster_DreamIDV(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                dreamidv_ckpt=dreamidv_ckpt,
            )
        else:
            dreamidv_ckpt = os.path.join(folder_paths.models_dir, 'DreamID-V', 'dreamidv.pth')
            print('use origin DreamID-V')
            from .dreamidv_wan import DreamIDV
            wan_swapface = DreamIDV(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                dreamidv_ckpt=dreamidv_ckpt,
            )
        return (wan_swapface, )

class RunningHub_DreamID_V_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"type": (["Wan2.2 I2V", "Wan2.1 T2V"], ),
                "pipeline": ("RH_DreamID-V_Pipeline", ),
                "video": ("VIDEO", ),
                "ref_image": ("IMAGE", ),
                "size": (["832*480", "1280*720", "480*832", "720*1280", "custom"], {"default": "832*480"}),
                "frame_num": ("INT", {"default": 81, "min": 1, 'step': 4}),
                "sample_steps": ("INT", {"default": 20,}),
                "fps": ("INT", {"default": 24,}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "face_detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'VIDEO')
    RETURN_NAMES = ('frames', 'video')
    FUNCTION = "sample"
    CATEGORY = "RunningHub/DreamID-V"

    # OUTPUT_NODE = True

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def create_video_with_audio(self, frames_tensor, fps, source_video_path, output_path):
        """
        Create video from frames tensor and copy audio from source video.
        
        Args:
            frames_tensor: Tensor of shape (N, H, W, C) with values in [0, 1]
            fps: Frames per second
            source_video_path: Path to source video for audio extraction
            output_path: Output video file path
        """
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        
        # Convert tensor to numpy frames
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        num_frames, height, width, channels = frames_np.shape
        
        # Write frames to temp video using cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_video_path}")
        
        for i in range(num_frames):
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"[DreamID-V] Wrote {num_frames} frames to temp video")
        
        # Check if source video has audio
        has_audio = False
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a:0', source_video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                if info.get('streams'):
                    has_audio = True
        except Exception as e:
            print(f"[DreamID-V] Could not probe audio: {e}")
        
        # Combine video with audio from source
        if has_audio:
            print(f"[DreamID-V] Copying audio from source video...")
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', source_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output_path
            ]
        else:
            print(f"[DreamID-V] No audio in source video, encoding video only...")
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                output_path
            ]
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if process.returncode != 0:
                print(f"[DreamID-V] FFmpeg error: {process.stderr}")
                raise RuntimeError(f"FFmpeg failed: {process.stderr}")
            print(f"[DreamID-V] Video created successfully: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video encoding timed out")
        finally:
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        return output_path

    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            # Fallback: return file path as string
            return video_path

    def frame_2_tensor(self, frame, target_w, target_h):
        resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.array(resized_frame).astype(np.float32) / 255.0)

    def sample(self, **kwargs):

        #kiki hardcode
        sample_shift = 5.0
        sample_solver = 'unipc'
        sample_guide_scale_img = 4.0

        pipeline = kwargs.get('pipeline')
        pipeline.config.sample_fps = kwargs.get('fps')
        sample_steps = kwargs.get('sample_steps')
        self.pbar = comfy.utils.ProgressBar(sample_steps + 1)
        ref_video_path = kwargs.get('video').get_stream_source()
        
        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        ref_image_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.png')
        ref_image.save(ref_image_path)
        size = kwargs.get('size')
        if size == 'custom':
            custom_width = kwargs.get('custom_width', 832)
            custom_height = kwargs.get('custom_height', 480)
            size_tuple = (custom_width, custom_height)
        else:
            from .dreamidv_wan.configs import SIZE_CONFIGS
            size_tuple = SIZE_CONFIGS[size]
        seed = kwargs.get('seed') ^ (2 ** 32)
        frame_num = kwargs.get('frame_num')

        face_detection_threshold = kwargs.get('face_detection_threshold', 0.5)
        try:
            detected_frames, pose_frames, mask_frames, skip_frames_index, skip_frames_data = generate_pose_and_mask_videos(
                ref_video_path=ref_video_path,
                ref_image_path=ref_image_path,
                face_detection_threshold=face_detection_threshold
            )
        except:
            raise ValueError("Pose and mask video generation failed. no pose detected in the reference video.")
        print(f'skip_frames_index: {skip_frames_index}')
        text_prompt = 'change face'

        # ref_paths = [
        #     ref_video_path,
        #     ref_mask_path,
        #     ref_image_path,
        #     ref_pose_path
        # ]
        ref_data = [
            detected_frames,
            pose_frames,
            mask_frames,
            ref_image_path,
        ]

        self.update()

        generated = pipeline.generate(
            text_prompt,
            ref_data,
            size=size_tuple,
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale_img=sample_guide_scale_img,
            seed=seed,
            update_fn=self.update)
        print(f'generated video shape: {generated.shape}')
        
        # Convert to frames tensor (N, H, W, C) with values in [0, 1]
        frames = (generated.clamp(-1, 1).cpu().permute(1, 2, 3, 0) + 1.0) / 2.0
        print(frames.shape)

        frames_list = list(torch.unbind(frames, dim=0))
        target_w, target_h = frames.shape[2], frames.shape[1]
        for i in skip_frames_index:
            if i < frame_num:
                frames_list.insert(i, self.frame_2_tensor(skip_frames_data[i], target_w, target_h))
        frames_list = frames_list[:frame_num]
        frames = torch.stack(frames_list, dim=0)
        # print(frames.shape)
        
        # Create output video with audio from source
        fps = kwargs.get('fps')
        output_dir = folder_paths.get_output_directory()
        output_filename = f"dreamidv_{uuid.uuid4()}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # self.create_video_with_audio(frames, fps, ref_video_path, output_path)
        self.create_video_with_audio(frames, fps, ref_video_path, output_path)
        
        # Create VIDEO object
        video_obj = self.create_video_object(output_path)
        
        return (frames, video_obj)

    def update(self):
        self.pbar.update(1)

class RunningHub_DreamID_V_Sampler_Test:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"type": (["Wan2.2 I2V", "Wan2.1 T2V"], ),
                # "pipeline": ("RH_DreamID-V_Pipeline", ),
                "video": ("VIDEO", ),
                "ref_image": ("IMAGE", ),
                "size": (["832*480", "1280*720", "480*832", "720*1280", "custom"], {"default": "832*480"}),
                "frame_num": ("INT", {"default": 81, "min": 1, 'step': 4}),
                "sample_steps": ("INT", {"default": 20,}),
                "fps": ("INT", {"default": 24,}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'VIDEO')
    RETURN_NAMES = ('frames', 'video')
    FUNCTION = "sample"
    CATEGORY = "RunningHub/DreamID-V"

    OUTPUT_NODE = True

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def sample(self, **kwargs):

        #kiki hardcode
        sample_shift = 5.0
        sample_solver = 'unipc'
        sample_guide_scale_img = 4.0
        fps = kwargs.get('fps')

        sample_steps = kwargs.get('sample_steps')
        self.pbar = comfy.utils.ProgressBar(sample_steps + 1)
        # ref_video_path = kwargs.get('video').get_stream_source()
        video_path = kwargs.get('video').get_stream_source()
        ref_video_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.mp4')
        # skip_frames_index, skip_frames_data = prehandle_video(video_path, ref_video_path, fps)
        # print(f'skip_frames_index: {skip_frames_index}')
        
        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        ref_image_path = os.path.join(folder_paths.get_temp_directory(), f'dreamidv_{uuid.uuid4()}.png')
        ref_image.save(ref_image_path)
        size = kwargs.get('size')

        detected_frames, pose_frames, mask_frames, skip_frames_index, skip_frames_data = generate_pose_and_mask_videos(
                # ref_video_path=ref_video_path,
                ref_video_path = video_path,
                ref_image_path=ref_image_path
            )
        # print(len(detected_frames))
        # print(len(pose_frames))
        # print(len(mask_frames))
        print(skip_frames_index)
        print(len(skip_frames_data))

        # frame_image = Image.fromarray(detected_frames[0])
        # print(frame_image.size)

        # import imageio
        # frames = imageio.get_reader(ref_video_path)
        # images = []
        # for i, frame in enumerate(frames):
        #     print(frame.shape)
        #     image = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
        #     images.append(image)
        # images = torch.stack(images)
        # print(images.shape)
        # frames_list = list(torch.unbind(images, dim=0))
        # for i in skip_frames_index:
        #     print(skip_frames_data[i].shape)
        #     frames_list.insert(i, torch.from_numpy(np.array(skip_frames_data[i]).astype(np.float32) / 255.0))
        # images = torch.stack(frames_list, dim=0)

        return (images, )


NODE_CLASS_MAPPINGS = {
    "RunningHub_DreamID-V_Loader": RunningHub_DreamID_V_Loader,
    "RunningHub_DreamID-V_Sampler": RunningHub_DreamID_V_Sampler,
    # "RunningHub_DreamID_V_Test": RunningHub_DreamID_V_Sampler_Test,
}