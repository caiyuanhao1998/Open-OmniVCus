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
s3_dir = 'yuanhao_cai/VideoEdit/label/single_cus_test/'

json_s3_dump_path = 'yuanhao_cai/VideoEdit/label/'

save_dir = 'index_jsons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


total_video_names = []

file_names = list_s3_files(bucket_name, s3_dir)
video_names = [f for f in file_names if f.endswith('.txt')]
total_video_names += video_names
    

# stx()

# 将所有的 video_names 存到一个 json 文件里
local_json_path = save_dir + 'single_cus_test.json'
with open(local_json_path, 'w') as file:
    json.dump(total_video_names, file, indent=4)

# 上传 json 文件到 s3
os.system(f'aws s3 cp {local_json_path} s3://{bucket_name}/{json_s3_dump_path}')