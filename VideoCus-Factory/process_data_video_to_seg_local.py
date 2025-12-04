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


'''
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
'''
    

# 这个 predictor 就是构造出来的 video predictor
# bboxes 就是 kosmos-2 的一个输出，是一个 list，每个元素是一个 dict，包含了一个 bounding box 的信息
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


# 处理视频的分割结果，生成两个帧的分割掩码
# video_segments 就只有两帧
# To-Do: 把这个改成 multi-frame ?
def process_segments(video_segments, frames_np_lst_selected):
    first_frame_segment = video_segments[0]
    # 创建一个与第一帧形状大小相同的全零掩码
    first_frame_mask = np.zeros_like(frames_np_lst_selected[0])[:,:,0]
    for i in range(len(first_frame_segment)):
        mask = first_frame_segment[i+1]
        first_frame_mask = first_frame_mask * (1-mask) + (first_frame_mask * 0 + 1 + i) * mask

    last_frame_segment = video_segments[1]
    last_frame_mask = np.zeros_like(frames_np_lst_selected[0])[:,:,0]
    for i in range(len(last_frame_segment)):
        mask = last_frame_segment[i+1]
        last_frame_mask = last_frame_mask * (1-mask) + (last_frame_mask * 0 + 1 + i) * mask 
    
    image1, image2 = frames_np_lst_selected
    mask1, mask2 = first_frame_mask * 20, last_frame_mask * 20
    return image1, image2, mask1, mask2


# frames_np_lst_selected 全要
def process_segments_video(video_segments, frames_np_lst_selected):
    mask_list = []
    for frame_segments in range(len(video_segments)):
        cur_frame_segment = video_segments[frame_segments]
        frame_mask = np.zeros_like(frames_np_lst_selected[frame_segments])[:,:,0]
        for i in range(len(cur_frame_segment)):
            mask = cur_frame_segment[i+1]
            frame_mask = frame_mask * (1-mask) + (frame_mask * 0 + 1 + i) * mask
        mask_list.append(frame_mask*20)

    return mask_list


# 在图像中心切一个方形区域
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



# kosmos-2 functional model
# 输入是一个 image numpy array
# 输出是 caption (整张图的 caption), entities (名词，对应图中的物体), boxes_real (entities 对应的 bboxes)
def cap_image(image_np):
    height, width, _ = image_np.shape
    image = Image.fromarray(image_np)
    prompt = "<grounding> An image of"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 下面这个 model 就是 kosmos-2 模型
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
kosmos2_weight_path = "/sensei-fs/users/yuanhaoc/pretrained_model/kosmos-2-patch14-224"
model = Kosmos2ForConditionalGeneration.from_pretrained(kosmos2_weight_path).to(device)
processor = AutoProcessor.from_pretrained(kosmos2_weight_path)


# ===  sam-2  ===
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


#part_indexes = [i for i in range(56,187)]

#part_indexes = [i for i in range(56,122)]
#part_indexes = [i for i in range(122,187)]
#part_indexes = [i for i in range(188,290)]


#part_indexes = [i for i in range(1,100)]
# part_indexes = [i for i in range(300,500)]      # 控制文件夹顺序
#part_indexes = [i for i in range(200,300)]
part_indexes = [11,]

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


def check_mask_size(mask):
    mask_bin = mask > 0
    mask_portion = mask_bin.sum() / np.ones_like(mask_bin).sum()
    #print(mask_bin.sum(), mask_portion)
    if mask_portion > 0.2 * 0.1 and mask_portion < 0.6 * 0.6 and is_foreground_touching_edges(mask_bin):
        return 1
    else:
        return 0
    
def is_foreground_touching_edges(mask):
    """
    判断二值掩码的前景是否与图像的边缘相接。
    
    :param mask: 二值掩码, 1-channel的 numpy 数组, 前景为1, 背景为0
    :return: 如果前景与任意边缘相接, 返回 True; 否则返回 False
    """
    num_edge = 0
    # 检查上边缘
    if np.any(mask[0, :] == 1):
        num_edge += 1
    
    # 检查下边缘
    if np.any(mask[-1, :] == 1):
        num_edge += 1
    
    # 检查左边缘
    if np.any(mask[:, 0] == 1):
        num_edge += 1
    
    # 检查右边缘
    if np.any(mask[:, -1] == 1):
        num_edge += 1
    
    # 如果没有边缘接触前景
    if num_edge > 1:
        return 0
    else:
        return 1



# 把 part_indexes 分成 N 份，取第 k 份
part_indexes = split_list(part_indexes, N, k) 

for part_i in part_indexes:
    # bucket_name = 'xic-data'
    # s3_root = 'data/multisubject/kosmos_sam_image_4s/'
    local_root = "sample_data/"                  # 这可以切换成 local dir 的形式
    dir_name = f'part_{part_i}/'
    save_root = 'sample_label/segmentation/'

    # s3_dir = s3_root + dir_name
    local_dir = local_root + dir_name
    save_dir = save_root + dir_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # image_names = list_s3_files(bucket_name, s3_dir)
    # image_names = os.listdir(local_dir)
    video_names = [f for f in os.listdir(local_dir) if f.endswith('.mp4')]
    # image_names = [i for i in image_names if 'img1' in i]
    # len(image_names) = 12800, 'data/multisubject/kosmos_sam_image_4s/part_300/14950_0_0_img1.jpg'


    # stx()

    for i in tqdm(range(len(video_names))):
        # try:
            json_dict = {}
            video_path = local_dir + video_names[i]
            name = video_path.split('/')[-1]
            save_path = save_dir  + name
            #image1 = Image.open(path1)  #read_image_from_s3(bucket_name, path1)
            #image2 = Image.open(path2)   #read_image_from_s3(bucket_name, path2)
            # image1 = read_image_from_s3(bucket_name, path1)
            # image2 = read_image_from_s3(bucket_name, path2)
            #frames_np_lst_selected = [ center_square_crop(np.array(image1)), center_square_crop(np.array(image2))]

            # 读取 .mp4 视频数据获取里面的帧
            video = cv2.VideoCapture(video_path)
            frames_np_lst = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frames_np_lst.append(frame)
            

            # 从 frames_np_lst 中随机取一帧
            # stx()
            idx_caption_frame = random.choice(range(len(frames_np_lst)))
            caption_frame = frames_np_lst[idx_caption_frame]

            # 此处只存 frame1 的 caption, entities, boxes
            # 用 SAM2 处理 frame2 的 masks 只是为了 filter, 确保两帧都有同样的物体
            # entity 在 caption 里面的开始和结束的位置
            '''
                caption [str]: 'An image of a sea turtle swimming in the ocean'
                entities [list of tuples]: [('a sea turtle', (12, 24), [(0.296875, 0.171875, 0.703125, 0.984375)]), ('the ocean', (37, 46), [(0.015625, 0.015625, 0.984375, 0.984375)])]
                boxes [list of np.array]: [array([ 74.,  34., 165., 211.], dtype=float32), array([  2.,   2., 165., 211.], dtype=float32)]
            '''
            caption, entities, boxes = cap_image(caption_frame)
            json_dict['caption'] = caption
            json_dict['entities'] = entities
            save_json_path = save_dir + name.replace('.mp4','.json')
            
            # stx()
            '''
                video_segments 是一个 dict, keys 为 [0, 1]
                video_segments[0] 也是一个 dict, keys 为 [1, 2].
                video_segments[0][1] 是一个 numpy array, shape 为 (384, 640), dtype 为 bool
                image1 - (384, 640, 3), mask1 - (384, 640) 这里的 mask 并不是 bool 值了
            '''
            video_segments = get_video_segments(predictor, frames_np_lst, boxes)
            # stx()
            mask_list = process_segments_video(video_segments, frames_np_lst)
            # stx()

            # 将 mask_list 存成一段 video
            video_mask_name = name.replace('.mp4', '_mask.mp4')
            video_mask_save_path = save_dir + video_mask_name
            with imageio.get_writer(video_mask_save_path, mode='I', fps=24) as video_mask:
                for mask in mask_list:
                    # 将 mask 写入 video_mask
                    video_mask.append_data(mask.astype(np.uint8))

            
            # stx()
            # ==============================
            #  保存 caption frame
            caption_frame_save_path = save_dir + name.replace('.mp4', '_caption_frame.jpg')
            cv2.imwrite(caption_frame_save_path, caption_frame)

            # stx()


            with open(save_json_path, 'w') as file:
                json.dump(json_dict, file, indent=4)  # indent=4 is optional, for pretty printing, indent 表示缩进
        # except:
        #     print('===== error ====')
        #     print(video_names[i])
