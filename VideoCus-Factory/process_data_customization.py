import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import imageio.v3 as iio
import imageio

from pdb import set_trace as stx


# =========================
# Utils
# =========================
def split_list(lst, N, K):
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N must be >0 and 1<=K<=N")
    length = len(lst)
    chunk_size = length // N
    remainder = length % N
    chunks, start = [], 0
    for i in range(N):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks[K - 1]


def is_foreground_touching_edges(mask):
    num_edge = 0
    if np.any(mask[0, :]): num_edge += 1
    if np.any(mask[-1, :]): num_edge += 1
    if np.any(mask[:, 0]): num_edge += 1
    if np.any(mask[:, -1]): num_edge += 1
    return 1 if num_edge <= 1 else 0


def check_mask_size(mask):
    mask = mask > 0
    portion = mask.sum() / mask.size
    return 1 if (0.01 < portion < 0.36 and is_foreground_touching_edges(mask)) else 0


# =========================
# Video IO (imageio)
# =========================
def read_video_local(video_path, max_frames=None, stride=1):
    """
    imageio read → RGB uint8
    """
    frames = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if i % stride != 0:
            continue
        frames.append(frame)  # already RGB uint8
        if max_frames is not None and len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return frames


def get_video_fps(video_path, default=8):
    try:
        meta = iio.immeta(video_path)
        return float(meta.get("fps", default))
    except Exception:
        return default


def write_mask_video(mask_rgb_list, save_path, fps):
    """
    mask_rgb_list: list of (H, W, 3) uint8 in {0,255}
    Save as mp4 using imageio.get_writer (lossy allowed).
    Black background, white foreground.
    """
    with imageio.get_writer(save_path, mode='I', fps=fps) as video_mask:
        for frame in mask_rgb_list:
            video_mask.append_data(frame)


# =========================
# SAM2 helpers
# =========================
def get_video_segments(predictor, frames, boxes):
    inference_state = predictor.init_state(video_path=None, video_frames=frames)
    predictor.reset_state(inference_state)

    for i, box in enumerate(boxes):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i + 1,
            box=box,
        )

    video_segments = {}
    for t, obj_ids, logits in predictor.propagate_in_video(inference_state):
        video_segments[t] = {
            obj_id: (logits[i, 0] > 0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }
    return video_segments


def process_segments_video(video_segments, frames):
    mask_list = []
    for t in range(len(video_segments)):
        frame_mask = np.zeros(frames[t].shape[:2], dtype=np.int32)
        for i, (_, mask) in enumerate(video_segments[t].items()):
            frame_mask[mask] = (i + 1)
        mask_list.append(frame_mask * 20)
    return mask_list


# =========================
# Kosmos-2
# =========================
def cap_image(image_np, processor, model, device):
    H, W, _ = image_np.shape
    image = Image.fromarray(image_np)
    prompt = "<grounding> An image of"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        max_new_tokens=64,
    )
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    caption, entities = processor.post_process_generation(text)

    boxes = []
    for ent in entities:
        x1, y1, x2, y2 = ent[2][0]
        boxes.append(np.array([x1*W, y1*H, x2*W, y2*H], dtype=np.float32))
    return caption, entities, boxes


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", type=int, required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--kosmos2_weight_path", type=str, default="checkpoints/kosmos-2-patch14-224")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Kosmos2ForConditionalGeneration.from_pretrained(args.kosmos2_weight_path).to(device)
    processor = AutoProcessor.from_pretrained(args.kosmos2_weight_path)

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    with open(args.input_json) as f:
        entries = json.load(f)

    entry_ids = split_list(list(range(len(entries))), args.N, args.k)

    json_stem = os.path.splitext(os.path.basename(args.input_json))[0]
    save_base = os.path.join(args.save_root, json_stem)
    os.makedirs(save_base, exist_ok=True)

    for idx in tqdm(entry_ids):
        entry = entries[idx]
        vid = entry["vid"]
        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue

        vid_dir = os.path.join(save_base, vid)
        os.makedirs(vid_dir, exist_ok=True)

        # =========================
        # NEW: save original prompt to prompt.txt
        # =========================
        original_prompt = entry.get("original", "")
        prompt_txt_path = os.path.join(vid_dir, "prompt.txt")
        with open(prompt_txt_path, "w", encoding="utf-8") as pf:
            pf.write(str(original_prompt).strip() + "\n")

        frames = read_video_local(video_path, args.max_frames, args.stride)
        fps = get_video_fps(video_path)

        t0 = random.randrange(len(frames))
        caption_frame = frames[t0]

        caption, entities, boxes = cap_image(caption_frame, processor, model, device)
        video_segments = get_video_segments(predictor, frames, boxes)
        mask_list = process_segments_video(video_segments, frames)

        common = set.intersection(*[set(np.unique(m)) for m in mask_list])
        common.discard(0)

        json_out = {
            "vid": vid,
            "caption": caption,
            "entities": entities,
            "common_entities": [],
        }

        for label in common:
            valid = all(
                check_mask_size((m == label).astype(np.uint8))
                for m in mask_list
            )
            if not valid:
                continue

            json_out["common_entities"].append(int(label))

            # entity png
            m0 = (mask_list[t0] == label).astype(np.uint8)
            ent = caption_frame * m0[:, :, None]
            iio.imwrite(os.path.join(vid_dir, f"entity_{label}.png"), ent)

            # entity mask video
            mask_rgb = [
                np.repeat(((m == label).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
                for m in mask_list
            ]
            write_mask_video(mask_rgb, os.path.join(vid_dir, f"entity_{label}.mp4"), fps)

        iio.imwrite(os.path.join(vid_dir, "caption_frame.png"), caption_frame)
        iio.imwrite(os.path.join(vid_dir, "mask.png"), mask_list[t0].astype(np.uint8))

        with open(os.path.join(vid_dir, "caption_frame.json"), "w") as f:
            json.dump(json_out, f, indent=2)


if __name__ == "__main__":
    main()