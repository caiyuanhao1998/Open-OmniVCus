# 将视频每一帧都取出来

video_path = '/sensei-fs/users/yuanhaoc/Video-Depth-Anything/outputs/original_video/1.mp4'
output_dir = 'outputs/frames'

import os

os.makedirs(output_dir, exist_ok=True)

import cv2

cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'{output_dir}/{count}.png', frame)
    count += 1
cap.release()
cv2.destroyAllWindows()