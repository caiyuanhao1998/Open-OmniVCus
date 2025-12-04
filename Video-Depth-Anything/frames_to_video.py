# 将文件夹中的图片合成一个视频

import cv2
import os

image_path = '/sensei-fs/users/yuanhaoc/Video-Depth-Anything/outputs/images_1'

image_list = os.listdir(image_path)

image_list.sort()

# 将图片合成一个视频 .mp4 文件
fps = 24
size = (256, 256)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('/sensei-fs/users/yuanhaoc/Video-Depth-Anything/outputs/1.mp4', fourcc, fps, size)

for i in range(120):
    img = cv2.imread(os.path.join(image_path, image_list[i]))
    video.write(img)

video.release()
cv2.destroyAllWindows()
