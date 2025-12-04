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




def get_video_segments(predictor, frames_np_lst_selected, boxes):
    inference_state = predictor.init_state(video_path=None, video_frames=frames_np_lst_selected)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with

    for i in range(len(boxes)):
        ann_obj_id = i + 1
        box = boxes[i]
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box = box
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i,0] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    return video_segments

    

def process_segments(video_segments, frames_np_lst_selected):
    first_frame_segment = video_segments[0]
    first_frame_mask = np.zeros_like(frames_np_lst_selected[0])[:,:,0]
    for i in range(len(first_frame_segment)):
        mask = first_frame_segment[i+1]
        first_frame_mask = first_frame_mask * (1-mask) + (first_frame_mask * 0 + 1 + i) * mask

    image1, image2 = frames_np_lst_selected
    mask1 = first_frame_mask * 20
    return image1, image2, mask1, mask1


def center_square_crop(image):
    # 获取原始图像的尺寸
    height, width = image.shape[:2]
    
    # 确定正方形的边长为短边的长度
    side_length = min(height, width)
    
    # 计算中心裁剪的起始位置
    x_start = (width - side_length) // 2
    y_start = (height - side_length) // 2
    
    # 裁剪图像
    cropped_image = image[y_start:y_start + side_length, x_start:x_start + side_length]
    
    return cropped_image




def cap_image(image_np):
    height, width, _ = image_np.shape
    image = Image.fromarray(image_np)
    prompt = "<grounding> An image of"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    caption, entities = processor.post_process_generation(generated_text)
    
    boxes_real = []
    
    for ent in entities:
        box = ent[2][0]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        boxes_real.append( np.array([x1,y1,x2,y2], dtype=np.float32))
        
    return caption, entities, boxes_real

# ===  kosmos-2  ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Kosmos2ForConditionalGeneration.from_pretrained("/sensei-fs/users/xic/data/weight/model/kosmos-2-patch14-224").to(device)
processor = AutoProcessor.from_pretrained("/sensei-fs/users/xic/data/weight/model/kosmos-2-patch14-224")


# ===  sam-2  ===
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)



part_indexes = [f"{i:03d}" for i in range(500)]
 

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


part_indexes = split_list(part_indexes, N, k)
local_root = "/mnt/localssd/openimage/images_sub/" 
dir_names = os.listdir(local_root)

for part_i in part_indexes:
    #bucket_name = 'xic-data'
    #s3_root = 'data/multisubject/kosmos_sam/'
    
    dir_name = part_i + '/'
    save_root = '/sensei-fs/users/xic/data/data/openimage/demo/'
    #save_root = '/mnt/localssd/openimage/images_mask/'

    #s3_dir = s3_root + dir_name
    local_dir = local_root + dir_name
    save_dir = save_root + dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #image_names = list_s3_files(bucket_name, s3_dir)
    image_names = os.listdir(local_dir)
    #image_names = [i for i in image_names if 'img1' in i]

    for i in tqdm(range(len(image_names))):
        #try:
            json_dict = {}
            path1 = local_dir + image_names[i]
            path2 = path1
            name1 = path1.split('/')[-1]
            name2 = path2.split('/')[-1]
            save_path1, save_path2 = save_dir  + name1, save_dir  + name2
            image1 = Image.open(path1)  #read_image_from_s3(bucket_name, path1)
            frames_np_lst_selected = [ np.array(image1), np.array(image1)]

            caption, entities, boxes = cap_image(frames_np_lst_selected[0])
            json_dict['caption'] = caption
            json_dict['entities'] = entities
            save_json_path = save_dir + name1.replace('.jpg','.json')
            
            video_segments = get_video_segments(predictor, frames_np_lst_selected, boxes)
            image1, image2, mask1, mask2 = process_segments(video_segments, frames_np_lst_selected)
            save_path1 = save_path1.replace('.jpg','.png')
            save_path2 = save_path2.replace('.jpg','.png')
            mask1 = np.stack([mask1,mask1,mask1],-1)
            mask2 = np.stack([mask2,mask2,mask2],-1)
            cv2.imwrite(save_path1, mask1)
            #cv2.imwrite(save_path2, mask2)
            with open(save_json_path, 'w') as file:
                json.dump(json_dict, file, indent=4)  # indent=4 is optional, for pretty printing
        #except:
        #    print('===== error ====')
        #    print(image_names[i])
