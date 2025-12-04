import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from data_utils import *
from tqdm import tqdm
import cv2
import json
import argparse

from pdb import set_trace as stx
import random
import imageio


# 遍历指定的 s3 路径，把里面的 .mp4 文件名都存到一个 json 里面
# 然后把这个 json 上传到 s3 上


bucket_name = 'cis-intern-2024'
# s3_dir = 'yuanhao_cai/VideoEdit/label/double_cus_test/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/double_cus_test_2/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/four_cus_test/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/four_cus_test_v2/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/four_cus_test_v3/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/depth_sincus_v1/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/seg_sincus_v1/'
# s3_dir = 'yuanhao_cai/VideoEdit/label/depth_sincus_v2/'
s3_dir = 'yuanhao_cai/VideoEdit/label/seg_sincus_v2/'

json_s3_dump_path = 'yuanhao_cai/VideoEdit/label/'

save_dir = 'index_jsons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 找出当前 s3_dir 下的所有子文件名，然后对每一个子文件进行遍历
folder_names = list_s3_folders(bucket_name, s3_dir)

total_video_names = []

for i in tqdm(range(len(folder_names))):

    folder_name = folder_names[i]
    file_names = list_s3_files_no_subfolders(bucket_name, folder_name)
    video_names = [f for f in file_names if f.endswith('.txt')]
    total_video_names += video_names
    # stx()
    

# stx()

# 将所有的 video_names 存到一个 json 文件里
# local_json_path = save_dir + 'depth_sincus.json'
local_json_path = save_dir + 'seg_sincus.json'
with open(local_json_path, 'w') as file:
    json.dump(total_video_names, file, indent=4)

# 上传 json 文件到 s3
os.system(f'aws s3 cp {local_json_path} s3://{bucket_name}/{json_s3_dump_path}')