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


bucket_name = 'yuqzhou-data'
original_json_path = 'MultiEdit/OmniEdit/combined/combined_metadata.json'


json_s3_dump_bucket = 'cis-intern-2024'
json_s3_dump_path = 'yuanhao_cai/VideoEdit/label/'

save_dir = 'index_jsons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取json文件
original_json = read_json_from_s3(bucket_name, original_json_path)

# stx()

'''
    len(original_json) = 1199285
    original_json[0].keys() - dict_keys(['omni_edit_id', 'task', 'edited_prompt_list', 'width', 'height', 'sc_score_1', 'sc_score_2', 'sc_reasoning', 'pq_score', 'pq_reasoning', 'o_score', 'src_img_path', 'edited_img_path'])
'''

# 遍历 original_json 剔除掉不符合要求的数据
filtered_json = []
for i in tqdm(range(len(original_json))):
    if original_json[i]['src_img_path'] == None or original_json[i]['edited_img_path'] == None:
        continue
    else:
        filtered_json.append(original_json[i])

print(f'The length of filtered data = {len(filtered_json)}')

# stx()

# 将所有的 video_names 存到一个 json 文件里
local_json_path = save_dir + 'Video_Instructive_Editing_Train.json'
with open(local_json_path, 'w') as file:
    json.dump(filtered_json, file, indent=4)
# 上传 json 文件到 s3
os.system(f'aws s3 cp {local_json_path} s3://{json_s3_dump_bucket}/{json_s3_dump_path}')