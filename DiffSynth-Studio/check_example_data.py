# 读取查看 vace training 的 example data 的数据类型
# 视频读写用 imageio, 图像用 cv2

import imageio
import cv2
import os
import numpy as np
from pdb import set_trace as stx

path = 'data/example_video_dataset/'

ref_img_name = 'reference_image.png'
ref_video_name = 'video1_softedge.mp4'
gt_video_name = 'video1.mp4'

'''
=== Reference Image ===
type: <class 'numpy.ndarray'>
dtype: uint8
shape (H, W, C): (1080, 1920, 3)
value range: 0 255

=== Reference Video ===
type: <class 'numpy.ndarray'>
dtype: uint8
shape (T, H, W, C): (176, 1088, 1920, 3)
value range: 0 255
unique value: 0 - 255 都有, 是连续值而非 binary value
三个 channel 是完全一样的, 相当于 copy 了三份

=== GT Video ===
type: <class 'numpy.ndarray'>
dtype: uint8
shape (T, H, W, C): (176, 1080, 1920, 3)
value range: 0 255
'''




# -----------------------------
# 1. 读取 reference image
# -----------------------------
ref_img_path = os.path.join(path, ref_img_name)
ref_img = cv2.imread(ref_img_path)  # BGR, uint8

assert ref_img is not None, f"Failed to load {ref_img_path}"

print("=== Reference Image ===")
print("type:", type(ref_img))
print("dtype:", ref_img.dtype)
print("shape (H, W, C):", ref_img.shape)
print("value range:", ref_img.min(), ref_img.max())

stx()

# -----------------------------
# 2. 读取 reference video（如 soft edge / condition video）
# -----------------------------
ref_video_path = os.path.join(path, ref_video_name)
ref_video_reader = imageio.get_reader(ref_video_path, format='ffmpeg')

ref_video_frames = []
for frame in ref_video_reader:
    # frame: (H, W, C), RGB, uint8
    ref_video_frames.append(frame)

ref_video_reader.close()

ref_video = np.stack(ref_video_frames, axis=0)  # (T, H, W, C)

print("=== Reference Video ===")
print("type:", type(ref_video))
print("dtype:", ref_video.dtype)
print("shape (T, H, W, C):", ref_video.shape)
print("value range:", ref_video.min(), ref_video.max())


stx()

# -----------------------------
# 3. 读取 GT video
# -----------------------------
gt_video_path = os.path.join(path, gt_video_name)
gt_video_reader = imageio.get_reader(gt_video_path, format='ffmpeg')

gt_video_frames = []
for frame in gt_video_reader:
    gt_video_frames.append(frame)

gt_video_reader.close()

gt_video = np.stack(gt_video_frames, axis=0)

print("=== GT Video ===")
print("type:", type(gt_video))
print("dtype:", gt_video.dtype)
print("shape (T, H, W, C):", gt_video.shape)
print("value range:", gt_video.min(), gt_video.max())


stx()

# -----------------------------
# 4. 简单一致性检查
# -----------------------------
print("=== Consistency Check ===")
print("Ref video frames:", ref_video.shape[0])
print("GT  video frames:", gt_video.shape[0])
print("Spatial size match:",
      ref_video.shape[1:3] == gt_video.shape[1:3])


stx()