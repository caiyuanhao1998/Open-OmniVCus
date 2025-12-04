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

bucket_name = 'cis-intern-2024'

s3_raw_data_dir = 'yuanhao_cai/VideoEdit/raw_data/'

s3_customization_dir = 'yuanhao_cai/VideoEdit/label/video_customize/'

s3_segmentation_dir = 'yuanhao_cai/VideoEdit/label/video2seg/'

s3_depth_dir = 'yuanhao_cai/VideoEdit/label/video2depth/'

s3_track_dir = 'yuanhao_cai/VideoEdit/label/video_motion_control/'

s3_vdepth_dir = 'yuanhao_cai/VideoEdit/label/video2vdepth/'


print("----------------------------------------- Count the number of raw data -----------------------------------------")
exist_raw_data_indexes = list_s3_folders('cis-intern-2024', s3_raw_data_dir)
exist_raw_data_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_raw_data_indexes])
print(f"Number of raw data folders: {len(exist_raw_data_id)}")
print(f"Number of raw data: {len(exist_raw_data_id)*50*32}")
print(f"Raw data id: {exist_raw_data_id}")

print("----------------------------------------- Count the number of customization -----------------------------------------")
exist_customization_indexes = list_s3_folders('cis-intern-2024', s3_customization_dir)
exist_customization_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_customization_indexes])
print(f"Number of customization folders: {len(exist_customization_id)}")
print(f"Number of customizations: {len(exist_customization_id)*50*32}")
print(f"Customization id: {exist_customization_id}")
print(f"Unprocessed customization id: {sorted(list(set(exist_raw_data_id) - set(exist_customization_id)))}")

print("----------------------------------------- Count the number of segmentation -----------------------------------------")
exist_segmentation_indexes = list_s3_folders('cis-intern-2024', s3_segmentation_dir)
exist_segmentation_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_segmentation_indexes])
print(f"Number of segmentation folders: {len(exist_segmentation_id)}")
print(f"Number of segmentations: {len(exist_segmentation_id)*50*32}")
print(f"Segmentation id: {exist_segmentation_id}")
print(f"Unprocessed segmentation id: {sorted(list(set(exist_raw_data_id) - set(exist_segmentation_id)))}")


print("-------------------------- Count the number of depth --------------------------")
exist_depth_indexes = list_s3_folders('cis-intern-2024', s3_depth_dir)
exist_depth_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_depth_indexes])
print(f"Number of depth folders: {len(exist_depth_id)}")
print(f"Number of depth: {len(exist_depth_id)*50*32}")
print(f"Depth id: {exist_depth_id}")
print(f"Unprocessed depth id: {sorted(list(set(exist_raw_data_id) - set(exist_depth_id)))}")


print("-------------------------- Count the number of track --------------------------")
exist_track_indexes = list_s3_folders('cis-intern-2024', s3_track_dir)
exist_track_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_track_indexes])
print(f"Number of track folders: {len(exist_track_id)}")
print(f"Number of track: {len(exist_track_id)*50*32}")
print(f"Track id: {exist_track_id}")
print(f"Unprocessed track id: {sorted(list(set(exist_raw_data_id) - set(exist_track_id)))}")


print("-------------------------- Count the number of video depth --------------------------")
exist_vdepth_indexes = list_s3_folders('cis-intern-2024', s3_vdepth_dir)
exist_vdepth_id = sorted([int(i.split('/')[-2].split('_')[-1]) for i in exist_vdepth_indexes])
print(f"Number of depth folders: {len(exist_vdepth_id)}")
print(f"Number of depth: {len(exist_vdepth_id)*50*32}")
print(f"Depth id: {exist_vdepth_id}")
print(f"Unprocessed depth id: {sorted(list(set(exist_raw_data_id) - set(exist_vdepth_id)))}")


    