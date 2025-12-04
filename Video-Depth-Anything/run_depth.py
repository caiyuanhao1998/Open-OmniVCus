# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

from pdb import set_trace as stx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)

    '''
        frames.shape - (70, 540, 960, 3)           value range - [0, 255]
        depths.shape - (70, 540, 960)              value range - [0, 1291.8812]
        target_fps - 24.0                          target_fps 
        fps - 24.0
    '''
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # stx()
    frame_idx = frames.shape[0] // 2
    # frame_idx = 149
    # frame_idx = 60
    # frame_idx = idx * video_length // 5
    frame = frames[frame_idx]
    depth = depths[frame_idx]

    img_output_dir = 'outputs/output_img'
    os.makedirs(img_output_dir, exist_ok=True)
    img_output_path = os.path.join(img_output_dir, f'{video_name}_{frame_idx}.png')
    depth_output_path = os.path.join(img_output_dir, f'{video_name}_{frame_idx}_depth.npy')

    import cv2
    cv2.imwrite(img_output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    d_min = depth.min()
    d_max = depth.max()
    depth = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    np.save(depth_output_path, depth)

    import matplotlib.cm as cm
    colormap = np.array(cm.get_cmap("inferno").colors)
    depth_vis = (colormap[depth] * 255).astype(np.uint8)
    depth_vis_output_path = os.path.join(img_output_dir, f'{video_name}_{frame_idx}_depth_vis.png')
    cv2.imwrite(depth_vis_output_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

    # stx()

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True)

    


