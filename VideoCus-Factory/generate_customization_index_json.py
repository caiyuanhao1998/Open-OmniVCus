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

json_s3_dump_path = 'yuanhao_cai/VideoEdit/label/'

save_dir = 'index_jsons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 找出当前 s3_dir 下的所有子文件名，然后对每一个子文件进行遍历
folder_names = list_s3_folders(bucket_name, s3_dir)

# print(folder_names)

total_caption_names = []
total_filter_mask_names = []
all_file_num = 0

# 对每个 folder 进行遍历
for i in tqdm(range(len(folder_names))):
    # 读取 s3_dir 下的所有文件名
    folder_name = folder_names[i]
    file_names = list_s3_files_no_subfolders(bucket_name, folder_name)
    all_file_num += len(file_names)
    # stx()
    # stx()
    filter_mask_names = [f for f in file_names if '_entity_' in f]


    caption_names = [(f.split('_entity_')[0] + '.json') for f in filter_mask_names]     # 这个 filter 规则保证了至少一个 entity
    # stx()
    total_caption_names += caption_names
    total_filter_mask_names += filter_mask_names

# stx()

print("number of data pairs: ", len(total_caption_names))
print("total number of files: ", all_file_num)
print("filter rato: ", len(total_caption_names) / all_file_num)


# 将所有的 video_names 存到一个 json 文件里

test_num = 200

local_json_path_1 = save_dir + 'video_customize_caption.json'
with open(local_json_path_1, 'w') as file:
    json.dump(total_caption_names[test_num:], file, indent=4)

local_json_path_1_test = save_dir + 'video_customize_caption_test.json'
with open(local_json_path_1_test, 'w') as file:
    json.dump(total_caption_names[:test_num], file, indent=4)

local_json_path_2 = save_dir + 'video_customize_entity.json'
with open(local_json_path_2, 'w') as file:
    json.dump(total_filter_mask_names[:test_num], file, indent=4)

local_json_path_2_test = save_dir + 'video_customize_entity_test.json'
with open(local_json_path_2_test, 'w') as file:
    json.dump(total_filter_mask_names[test_num:], file, indent=4)

# 上传 json 文件到 s3
os.system(f'aws s3 cp {local_json_path_1} s3://{bucket_name}/{json_s3_dump_path}')
os.system(f'aws s3 cp {local_json_path_1_test} s3://{bucket_name}/{json_s3_dump_path}')
os.system(f'aws s3 cp {local_json_path_2} s3://{bucket_name}/{json_s3_dump_path}')
os.system(f'aws s3 cp {local_json_path_2_test} s3://{bucket_name}/{json_s3_dump_path}')