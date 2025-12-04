import cv2
import numpy as np
from pdb import set_trace as stx
import imageio

# Read the video file
# path = '/mnt/localssd/yuanhaoc/video2seg/part_112/5612_0_19_mask.mp4'
# path = '/mnt/localssd/yuanhaoc/video2seg/part_506/6550_0_0_mask.mp4'
path = '/mnt/localssd/yuanhaoc/video_customize/part_0/0_0_22_mask.png'

# 读取灰度图像, unchanged 保持原始图像的通道数
mask_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# 转成 numpy array
mask_array = np.array(mask_image)

print(np.unique(mask_array))

stx()
