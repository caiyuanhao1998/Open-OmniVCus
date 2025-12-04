import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
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
parser.add_argument('--input_size', type=int, default=518)
parser.add_argument('--max_res', type=int, default=1280)
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
parser.add_argument('--target_fps', type=int, default=24, help='target fps of the input video, -1 means the original fps')
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
    

############################  load video depth anything  ###########################
from video_depth_anything.video_depth import VideoDepthAnything
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
video_depth_anything = video_depth_anything.to(device).eval()
####################################################################################





#part_indexes = [i for i in range(56,187)]

#part_indexes = [i for i in range(56,122)]
#part_indexes = [i for i in range(122,187)]
#part_indexes = [i for i in range(188,290)]


#part_indexes = [i for i in range(1,100)]
# part_indexes = [i for i in range(300,500)]      # 控制文件夹顺序
#part_indexes = [i for i in range(200,300)]
# part_indexes = [i for i in range(0,175)]

# cus_idx = 9

# all_part_indexes = [i for i in range((cus_idx-1)*100, cus_idx*100)]

# all_part_indexes = [i for i in range(200, 1000)]

# all_part_indexes = [i for i in range(0, 400)]

# all_part_indexes = [i for i in range(30, 800)]     # vdepth-1
# all_part_indexes = [i for i in range(800, 1050)]     # vdepth-2
# all_part_indexes = [i for i in range(1050, 1250)]     # vdepth-3
all_part_indexes = [i for i in range(170, 1070)]     # vdepth-1

s3_cus_label_dir = 'yuanhao_cai/VideoEdit/label/video2vdepth/'

exist_part_indexes = list_s3_folders('cis-intern-2024', s3_cus_label_dir)

exist_folders = [int(i.split('/')[-2].split('_')[-1]) for i in exist_part_indexes]

part_indexes = [i for i in all_part_indexes if i not in exist_folders]

print("number of part is", len(part_indexes))

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
    bucket_name = 'cis-intern-2024'
    s3_root = 'yuanhao_cai/VideoEdit/raw_data/'
    dir_name = f'part_{part_i}/'
    save_root = '/mnt/localssd/yuanhaoc/video2vdepth/'

    s3_dir = s3_root + dir_name
    save_dir = save_root + dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names = list_s3_files(bucket_name, s3_dir)
    # image_names = os.listdir(local_dir)
    # 对 filenames 进行 filter, 只保留 .mp4 文件
    video_names = [f for f in file_names if f.endswith('.mp4')]


    # stx()

    for i in tqdm(range(len(video_names))):
        try:
            json_dict = {}
            video_path = video_names[i]
            name = video_path.split('/')[-1]
            save_path = save_dir + name
            
            video_depth_name = name.replace('.mp4', '_depth.mp4')
            video_depth_save_path = save_dir + video_depth_name

            # 读取 .mp4 视频数据获取里面的帧
            # 创建一个新的 video writer，对读取的每一帧，将其转成 depth，然后写入新的 video
            frames_np_lst = read_video_from_s3(bucket_name, video_path)
            H, W, _ = frames_np_lst[0].shape
            frame_idx = 0
            frames = np.array(frames_np_lst).astype(np.uint8)           # (128, 384, 640, 3)
            # stx()
            depths, fps = video_depth_anything.infer_video_depth(frames, target_fps=24, input_size=args.input_size, device=device)

            # 将 depths 写入 video_depth_save_path
            with imageio.get_writer(video_depth_save_path, mode='I', fps=24) as video_depth:
                for i in range(len(depths)):
                    depth = depths[i]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    depth = depth * 255
                    video_depth.append_data(depth.astype(np.uint8))

            # 将所有的 frame 转成 depth
        except:
            print('===== error ====')
            print(video_names[i])