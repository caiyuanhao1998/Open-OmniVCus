import json
from PIL import Image
import boto3
from io import BytesIO
import botocore 
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
import time
import re
import random
import tempfile

import cv2

def read_npy_from_s3(bucket_name, key):
    """
    从 S3 读取 .npy 文件并加载为一个 numpy 数组。

    参数：
        bucket_name (str): S3 的桶名。
        key (str): .npy 文件的键（路径）。

    返回：
        data (numpy.ndarray): 加载的 .npy 文件数据。
    """
    # 从 S3 获取对象
    obj = objClient.get_object(Bucket=bucket_name, Key=key)["Body"]
    
    # 将对象写入临时文件后再加载
    with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as temp_npy:
        temp_npy.write(obj.read())
        temp_npy.flush()  # 确保数据写入完成

        # 使用 numpy.load 加载数组
        data = np.load(temp_npy.name, allow_pickle=True)

    return data

def read_video_from_s3(bucket_name, key):
    """
    从 S3 读取视频文件并解码为帧。

    参数：
        bucket_name (str): S3 的桶名。
        key (str): 视频文件的键（路径）。

    返回：
        frames (list): 包含视频帧的列表，每一帧为一个 numpy 数组。
    """
    # 下载视频到临时文件
    obj = objClient.get_object(Bucket=bucket_name, Key=key)["Body"]
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(obj.read())
        temp_video.flush()

        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(temp_video.name)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

    return frames


objClient = boto3.session.Session().client(
    service_name="s3",
    config=botocore.config.Config(
        region_name="us-west-2",
        connect_timeout=10,
        read_timeout=10,
        retries={"mode": "standard", "max_attempts": 10},
        signature_version=botocore.UNSIGNED,
    ),
)


def read_image_from_s3(bucket_name, key):
    image_s3 = objClient.get_object(Bucket=bucket_name, Key=key)["Body"]
    image = Image.open(image_s3).convert('RGB')
    return image


def list_s3_folders(bucket_name, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    folders = []
    for page in result:
        if 'CommonPrefixes' in page:
            for obj in page['CommonPrefixes']:
                folders.append(obj['Prefix'])
    return folders


def list_s3_files(bucket_name, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    files = []
    for page in result:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    return files


def list_s3_files_no_subfolders(bucket_name, prefix):
    """
    只列出 S3 上 prefix 目录下的文件，不遍历子文件夹。
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    # 关键是加上 Delimiter='/'，这样 S3 返回结果会将子文件夹单独放在 CommonPrefixes 里
    result = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter='/'
    )

    files = []
    for page in result:
        # 'Contents' 下就是 prefix 目录下（不含子目录）列出的文件
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # 如果 key 本身以 "/" 结尾，说明是一个目录而非文件，这里可直接略过
                if key.endswith('/'):
                    continue
                # 也可以根据需要做更多判断，比如只要 prefix 之后没有再包含 '/'
                relative_path = key[len(prefix):]
                if '/' not in relative_path:
                    files.append(key)
    
    return files


def read_json_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read().decode('utf-8')
    json_data = json.loads(data)
    return json_data
