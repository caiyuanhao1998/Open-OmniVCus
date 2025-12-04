# 将文件夹中的图片合成一个视频

import cv2
import os
import videoio
import numpy as np

image_path = '/sensei-fs/users/yuanhaoc/Video-Depth-Anything/outputs/images_0'
image_list = os.listdir(image_path)
image_list.sort()

# 将图片合成一个视频 .mp4 文件
# 使用 cv2 写入视频

def imageseq2video(images, filename, fps=24):
    # if images is uint8, convert to float32
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    videoio.videosave(filename, images, lossless=True, preset="veryfast", fps=fps)

# 使用 imageseq2video 创建视频
images = []
for i in range(120):
    img = cv2.imread(os.path.join(image_path, image_list[i]))
    # RGB 转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
images = np.array(images)

imageseq2video(images, '/sensei-fs/users/yuanhaoc/Video-Depth-Anything/outputs/0.mp4', fps=24)
