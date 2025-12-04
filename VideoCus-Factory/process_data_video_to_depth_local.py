import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from data_utils import *
from tqdm import tqdm
import cv2
import json
import argparse

from pdb import set_trace as stx
import random
import imageio




parser = argparse.ArgumentParser(description="An example program with argparse")
parser.add_argument("-G", type=int, required=True, help="Value of parameter a")
parser.add_argument("-k", type=int, required=True, help="Value of parameter b")
parser.add_argument("-N", type=int, required=True, help="Value of parameter b")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
k, N = args.k, args.N




# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
    

##########################  load depth anything v2  ##########################
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device)
##############################################################################





#part_indexes = [i for i in range(56,187)]

#part_indexes = [i for i in range(56,122)]
#part_indexes = [i for i in range(122,187)]
#part_indexes = [i for i in range(188,290)]


#part_indexes = [i for i in range(1,100)]
# part_indexes = [i for i in range(300,500)]      # 控制文件夹顺序
#part_indexes = [i for i in range(200,300)]
part_indexes = [11,]

def split_list(lst, N, K):
    """
    将列表 lst 按顺序分成 N 等份，并返回第 K 分。
    
    :param lst: 待分割的列表
    :param N: 要分成的份数
    :param K: 返回的第几份 (从 1 开始)
    :return: 第 K 份列表
    """
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N 必须大于 0 且 K 必须在 1 和 N 之间")
    
    length = len(lst)
    chunk_size = length // N
    remainder = length % N
    
    # 计算每个分块的起始和结束索引
    chunks = []
    start = 0
    for i in range(N):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks[K-1]



# 把 part_indexes 分成 N 份，取第 k 份
part_indexes = split_list(part_indexes, N, k) 

for part_i in part_indexes:
    # bucket_name = 'xic-data'
    # s3_root = 'data/multisubject/kosmos_sam_image_4s/'
    local_root = "sample_data/"                  # 这可以切换成 local dir 的形式
    dir_name = f'part_{part_i}/'
    save_root = 'sample_label/depth/'

    # s3_dir = s3_root + dir_name
    local_dir = local_root + dir_name
    save_dir = save_root + dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # image_names = list_s3_files(bucket_name, s3_dir)
    # image_names = os.listdir(local_dir)
    video_names = [f for f in os.listdir(local_dir) if f.endswith('.mp4')]
    # image_names = [i for i in image_names if 'img1' in i]
    # len(image_names) = 12800, 'data/multisubject/kosmos_sam_image_4s/part_300/14950_0_0_img1.jpg'


    # stx()

    for i in tqdm(range(len(video_names))):
        # try:
            json_dict = {}
            video_path = local_dir + video_names[i]
            name = video_path.split('/')[-1]
            save_path = save_dir + name
            
            video_depth_name = name.replace('.mp4', '_depth.mp4')
            video_depth_save_path = save_dir + video_depth_name

            # 读取 .mp4 视频数据获取里面的帧
            # 创建一个新的 video writer，对读取的每一帧，将其转成 depth，然后写入新的 video
            video = cv2.VideoCapture(video_path)
            frame_idx = 0
            with imageio.get_writer(video_depth_save_path, mode='I', fps=24) as video_depth:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx == 1:
                        H, W, _ = frame.shape
                    # Convert the frame to a PIL image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    inputs = image_processor(images=frame, return_tensors="pt")
                    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                    outputs = model(**inputs)
                    post_processed_output = image_processor.post_process_depth_estimation(outputs,target_sizes=[(H,W)])
                    predicted_depth = post_processed_output[0]["predicted_depth"]
                    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
                    depth = depth.detach().cpu().numpy() * 255
                    # 将 frame_depth 写入 video_depth
                    video_depth.append_data(depth.astype(np.uint8))

            # 将所有的 frame 转成 depth

        # except:
        #     print('===== error ====')
        #     print(video_names[i])
