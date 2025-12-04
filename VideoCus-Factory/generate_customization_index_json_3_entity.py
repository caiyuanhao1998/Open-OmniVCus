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


# 遍历指定的 s3 路径，把里面的 .mp4 文件名都存到一个 json 里面
# 然后把这个 json 上传到 s3 上


bucket_name = 'cis-intern-2024'
s3_dir = 'yuanhao_cai/VideoEdit/label/video_customize/'
original_json_file_path = 'yuanhao_cai/VideoEdit/label/video_customize_caption.json'
json_s3_dump_path = 'yuanhao_cai/VideoEdit/label/'
save_dir = 'index_jsons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


json_list = read_json_from_s3(bucket_name, original_json_file_path)



# print(folder_names)

total_caption_names = []
total_filter_mask_names = []
all_file_num = 0

# 对每个 folder 进行遍历
for i in tqdm(range(len(json_list))):
    # 读取 s3_dir 下的所有文件名
    cur_json_path = json_list[i]
    cur_json = read_json_from_s3(bucket_name, cur_json_path)

    if len(cur_json['common_entities']) >= 3:
        total_caption_names.append(cur_json_path)
        print(f"total num:{len(total_caption_names)}")
    
    # if len(total_caption_names) >= 100:
    #     break

# stx()

print("number of data pairs: ", len(total_caption_names))


# 将所有的 video_names 存到一个 json 文件里

test_num = 100

label = 'three_entity'

local_json_path_1 = save_dir + f'video_customize_caption_{label}.json'
with open(local_json_path_1, 'w') as file:
    json.dump(total_caption_names[test_num:], file, indent=4)

local_json_path_1_test = save_dir + f'video_customize_caption_test_{label}.json'
with open(local_json_path_1_test, 'w') as file:
    json.dump(total_caption_names[:test_num], file, indent=4)

# 上传 json 文件到 s3
os.system(f'aws s3 cp {local_json_path_1} s3://{bucket_name}/{json_s3_dump_path}')
os.system(f'aws s3 cp {local_json_path_1_test} s3://{bucket_name}/{json_s3_dump_path}')