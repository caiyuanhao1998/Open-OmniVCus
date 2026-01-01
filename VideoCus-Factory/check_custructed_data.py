# 读取查看 vace training 的 example data 的数据类型
# 视频读写用 imageio, 图像用 cv2

import imageio
import cv2
import os
import numpy as np
from pdb import set_trace as stx

path = '/genai/fsx-project/yuanhaoc/Open-OmniVCus/VideoCus-Factory/T2V_data/label/vidgen_filter_gemini_part_0_dpo_0/HTAfzgGB-Bs-Scene-0013'

ref_img_name = 'entity_40.png'
ref_video_name = 'entity_40.mp4'
mask_name = 'mask.png'




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
print("unique values:", np.unique(ref_video[0]))

stx()


# -----------------------------
# 3. 读取 mask image
# -----------------------------
mask_img_path = os.path.join(path, mask_name)
mask = cv2.imread(mask_img_path)  # BGR, uint8

assert ref_img is not None, f"Failed to load {mask_img_path}"

print("=== Mask Image ===")
print("type:", type(mask))
print("dtype:", mask.dtype)
print("shape (H, W, C):", mask.shape)
print("value range:", mask.min(), mask.max())
print("unique values:", np.unique(mask))

stx()