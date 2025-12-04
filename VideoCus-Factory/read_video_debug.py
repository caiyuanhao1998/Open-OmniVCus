import cv2
import numpy as np
from pdb import set_trace as stx
import imageio

# Read the video file
# path = '/mnt/localssd/yuanhaoc/video2seg/part_112/5612_0_19_mask.mp4'
# path = '/mnt/localssd/yuanhaoc/video2seg/part_506/6550_0_0_mask.mp4'
path = '/mnt/localssd/yuanhaoc/video2depth/part_175/17_0_19_depth.mp4'

# # cv2 来读取视频
# cap = cv2.VideoCapture(path)
# # 读取视频帧转为 numpy array
# frame_list = []
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_list.append(frame)

# # 将视频帧转为 numpy array
# frame_array = np.array(frame_list)


# 使用 imageio 来读取呢？
reader = imageio.get_reader(path)
frame_list = []
for i, frame in enumerate(reader):
    frame_list.append(frame)
frame_array = np.array(frame_list)

# print(np.unique(frame_array))

# stx()
