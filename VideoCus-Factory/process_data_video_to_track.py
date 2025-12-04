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
    

##########################  load cotracker3  ##########################
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
##############################################################################





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

all_part_indexes = [i for i in range(1156, 1690)]           # track-1

s3_cus_label_dir = 'yuanhao_cai/VideoEdit/label/video_motion_control/'

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
    save_root = '/mnt/localssd/yuanhaoc/video_motion_control/'

    s3_dir = s3_root + dir_name
    save_dir = save_root + dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names = list_s3_files(bucket_name, s3_dir)
    # image_names = os.listdir(local_dir)
    # 对 filenames 进行 filter, 只保留 .mp4 文件
    video_names = [f for f in file_names if f.endswith('.mp4')]

    # print(f"number of videos is {len(video_names)}")


    # stx()

    for i in tqdm(range(len(video_names))):
        try:
            json_dict = {}
            video_path = video_names[i]
            name = video_path.split('/')[-1]
            save_path = save_dir + name
            
            video_track_name = name.replace('.mp4', '_track.npy')
            video_track_save_path = save_dir + video_track_name

            video_track_vis_name = name.replace('.mp4', '_track_vis.npy')
            video_track_vis_save_path = save_dir + video_track_vis_name

            # 读取 .mp4 视频数据获取里面的帧
            # 创建一个新的 video writer，对读取的每一帧，将其转成 depth，然后写入新的 video
            # frames_np_lst 是长度为 128 的 list, list 中每个元素的 shape 是 (384, 640, 3)
            frames_np_lst = read_video_from_s3(bucket_name, video_path)
            H, W, _ = frames_np_lst[0].shape

            # 将 frames_np_lst 转成 numpy 再转成 tensor
            grid_size = 10
            '''
                video shape: (1, 128, 3, 384, 640)
                pred_tracks shape: (1, 128, 100, 2)
                pred_visibility shape: (1, 128, 100)
            '''
            video = torch.tensor(np.array(frames_np_lst)).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
            pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1

            # 将 pred_tracks 和 pred_visibility 存成 numpy
            pred_tracks = pred_tracks[0].cpu().numpy()
            pred_visibility = pred_visibility[0].cpu().numpy()

            # 保存 pred_tracks 和 pred_visibility
            np.save(video_track_save_path, pred_tracks)
            np.save(video_track_vis_save_path, pred_visibility)

            # stx()

        except:
            print('===== error ====')
            print(video_names[i])
