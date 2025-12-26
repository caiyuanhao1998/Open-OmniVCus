import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import imageio.v3 as iio
import imageio

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


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


# =========================
# Video IO (imageio)
# =========================
def read_video_local(video_path, max_frames=None, stride=1):
    """
    imageio read → list of RGB uint8 frames
    """
    frames = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if i % stride != 0:
            continue
        # frame: (H,W,3) uint8, already RGB for most decoders
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return frames


def get_video_fps(video_path, default=24):
    try:
        meta = iio.immeta(video_path)
        return float(meta.get("fps", default))
    except Exception:
        return default


def write_gray_video(gray_list, save_path, fps):
    """
    gray_list: list of (H, W) uint8
    Write as mp4 (grayscale) via imageio.get_writer.
    """
    with imageio.get_writer(save_path, mode="I", fps=fps) as w:
        for g in gray_list:
            w.append_data(g)


# =========================
# Depth Anything v2
# =========================
@torch.inference_mode()
def depth_of_frame(frame_rgb_uint8, image_processor, model, device, out_hw):
    """
    frame_rgb_uint8: (H,W,3) uint8
    out_hw: (H,W) target size for depth map
    return: (H,W) uint8 depth (0..255)
    """
    frame_pil = Image.fromarray(frame_rgb_uint8)
    inputs = image_processor(images=frame_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    post = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[out_hw],  # [(H,W)]
    )
    predicted_depth = post[0]["predicted_depth"]  # (H,W) float tensor

    dmin = predicted_depth.min()
    dmax = predicted_depth.max()
    denom = (dmax - dmin)
    if denom.abs().item() < 1e-8:
        depth = torch.zeros_like(predicted_depth)
    else:
        depth = (predicted_depth - dmin) / denom

    depth_u8 = (depth * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return depth_u8


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", type=int, required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)

    # 仿照你第一份脚本的 IO 参数
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)

    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--default_fps", type=float, default=24)

    # Depth Anything v2
    parser.add_argument(
        "--depth_anything_repo",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    image_processor = AutoImageProcessor.from_pretrained(args.depth_anything_repo)
    model = AutoModelForDepthEstimation.from_pretrained(args.depth_anything_repo).to(device)
    model.eval()

    with open(args.input_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # entries 可以是 list[dict]，每个 dict 至少包含 vid
    entry_ids = split_list(list(range(len(entries))), args.N, args.k)

    json_stem = os.path.splitext(os.path.basename(args.input_json))[0]
    save_base = os.path.join(args.save_root, json_stem)
    os.makedirs(save_base, exist_ok=True)

    for idx in tqdm(entry_ids, desc=f"rank {args.k}/{args.N}"):
        entry = entries[idx]

        # 你 json 里如果不是 "vid"，这里改一下
        vid = entry.get("vid", None)
        if vid is None:
            continue

        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue

        vid_dir = os.path.join(save_base, str(vid))
        os.makedirs(vid_dir, exist_ok=True)

        # 保存原 prompt（如果存在）
        original_prompt = entry.get("original", entry.get("prompt", ""))
        with open(os.path.join(vid_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
            pf.write(str(original_prompt).strip() + "\n")

        try:
            frames = read_video_local(video_path, args.max_frames, args.stride)
            H, W, _ = frames[0].shape
            fps = get_video_fps(video_path, default=args.default_fps)

            depth_frames = []
            for fr in frames:
                depth_u8 = depth_of_frame(
                    fr,
                    image_processor=image_processor,
                    model=model,
                    device=device,
                    out_hw=(H, W),
                )
                depth_frames.append(depth_u8)

            # 写 depth mp4（灰度）
            out_depth_mp4 = os.path.join(vid_dir, "depth.mp4")
            write_gray_video(depth_frames, out_depth_mp4, fps=fps)

            # 可选：保存一张示例帧
            iio.imwrite(os.path.join(vid_dir, "depth_frame0.png"), depth_frames[0])

            # 记录一些 meta
            meta = {
                "vid": vid,
                "src_video": video_path,
                "depth_video": out_depth_mp4,
                "fps": fps,
                "num_frames": len(depth_frames),
                "stride": args.stride,
                "max_frames": args.max_frames,
                "depth_anything_repo": args.depth_anything_repo,
            }
            with open(os.path.join(vid_dir, "depth_meta.json"), "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)

        except Exception as e:
            print("===== error ====")
            print("vid:", vid)
            print("video_path:", video_path)
            print("err:", repr(e))


if __name__ == "__main__":
    main()