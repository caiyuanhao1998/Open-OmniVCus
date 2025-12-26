# 读取查看 vace training 的 example data 的数据类型
# 视频读写用 imageio, 图像用 cv2

import imageio
import cv2
import os
import numpy as np
from pdb import set_trace as stx




# -----------------------------
# 1. 读取 depth video
# -----------------------------
depth_video_path = '/genai/fsx-project/yuanhaoc/Open-OmniVCus/Video-Depth-Anything/T2V_data/label/vidgen_filter_gemini_part_0_dpo_0/HTAfzgGB-Bs-Scene-0013/depth.mp4'
depth_video_reader = imageio.get_reader(depth_video_path, format='ffmpeg')

depth_video_frames = []
for frame in depth_video_reader:
    # frame: (H, W, C), RGB, uint8
    depth_video_frames.append(frame)

depth_video_reader.close()

# 长为 224 的 list of dtype('unit8') - shape (720, 1280, 3)
depth_video = np.stack(depth_video_frames, axis=0)

print("=== Depth Video ===")
print("type:", type(depth_video))
print("dtype:", depth_video.dtype)
print("shape (T, H, W, C):", depth_video.shape)
print("value range:", depth_video.min(), depth_video.max())
print("unique values:", np.unique(depth_video[0]))


stx()

# -----------------------------
# 2. 读取 original video
# -----------------------------
original_video_path = '/genai/fsx-project/yuanhaoc/Open-OmniVCus/VideoCus-Factory/T2V_data/VIDGEN-1M_unzip/HTAfzgGB-Bs-Scene-0013.mp4'
original_video_reader = imageio.get_reader(original_video_path, format='ffmpeg')

original_video_frames = []
for frame in original_video_reader:
    # frame: (H, W, C), RGB, uint8
    original_video_frames.append(frame)

original_video_reader.close()

# stx()

original_video = np.stack(original_video_frames, axis=0)  # (T, H, W, C)

print("=== Original Video ===")
print("type:", type(original_video))
print("dtype:", original_video.dtype)
print("shape (T, H, W, C):", original_video.shape)
print("value range:", original_video.min(), original_video.max())
print("unique values:", np.unique(original_video[0]))

stx()



# -----------------------------
# 3. 读取 Example Depth Video
# -----------------------------
depth_video_path = '/genai/fsx-project/yuanhaoc/Open-OmniVCus/DiffSynth-Studio/data/examples/wan/depth_video.mp4'
depth_video_reader = imageio.get_reader(depth_video_path, format='ffmpeg')

depth_video_frames = []
for frame in depth_video_reader:
    # frame: (H, W, C), RGB, uint8
    depth_video_frames.append(frame)

depth_video_reader.close()

depth_video = np.stack(depth_video_frames, axis=0)

print("=== Depth Video ===")
print("type:", type(depth_video))
print("dtype:", depth_video.dtype)
print("shape (T, H, W, C):", depth_video.shape)
print("value range:", depth_video.min(), depth_video.max())
print("unique values:", np.unique(depth_video[0]))

stx()