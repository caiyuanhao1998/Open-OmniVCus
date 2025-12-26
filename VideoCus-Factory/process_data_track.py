import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as iio

from pdb import set_trace as stx


# =========================
# Utility functions
# =========================
def split_list(lst, N, K):
    """
    Split a list into N contiguous chunks and return the K-th chunk (1-indexed).
    """
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


def read_video_local(video_path, max_frames=None, stride=1):
    """
    Read a local video using imageio.

    Returns:
        frames: list of (H, W, 3) uint8 RGB frames
    """
    frames = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if stride > 1 and (i % stride != 0):
            continue
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return frames


def safe_makedirs(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def build_video_tensor(frames_np_lst, device):
    """
    Convert a list of RGB frames to a CoTracker-compatible tensor.

    Args:
        frames_np_lst: list of (H, W, 3) uint8
    Returns:
        video tensor of shape (1, T, 3, H, W), float32
    """
    arr = np.asarray(frames_np_lst)  # (T, H, W, 3)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float()
    return video.to(device)


# =========================
# Main
# =========================
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    # Distributed / GPU arguments
    parser.add_argument("-G", type=int, required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)

    # Input / output
    parser.add_argument(
        "--input_json", type=str, default=None,
        help="JSON list. Each entry must contain either `vid` or `video`."
    )
    parser.add_argument(
        "--video_root", type=str, required=True,
        help="Root directory of local mp4 files"
    )
    parser.add_argument(
        "--save_root", type=str, required=True,
        help="Root directory for outputs"
    )

    # Video loading (default: full video for exact frame alignment)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # CoTracker options
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="cotracker3_offline")

    # Optional: iterate over part_x folders
    parser.add_argument("--use_parts", action="store_true")
    parser.add_argument("--part_start", type=int, default=1156)
    parser.add_argument("--part_end", type=int, default=1690)
    parser.add_argument("--skip_if_exist", action="store_true")

    # Safety flag
    parser.add_argument(
        "--allow_mismatch", action="store_true",
        help="Allow track length to differ from original video frame count (NOT recommended)"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load CoTracker
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker", args.model_name
    ).to(device).eval()

    # ------------------------------------------------------------
    # Build task list
    # ------------------------------------------------------------
    tasks = []

    if args.use_parts:
        all_parts = list(range(args.part_start, args.part_end))
        part_list = split_list(all_parts, args.N, args.k)

        for part_i in part_list:
            part_dir = os.path.join(args.video_root, f"part_{part_i}")
            if not os.path.isdir(part_dir):
                continue
            for fn in sorted(os.listdir(part_dir)):
                if fn.endswith(".mp4"):
                    stem = os.path.splitext(fn)[0]
                    out_dir = os.path.join(args.save_root, f"part_{part_i}", stem)
                    tasks.append({
                        "video_path": os.path.join(part_dir, fn),
                        "out_dir": out_dir,
                        "stem": stem,
                    })
    else:
        if args.input_json is None:
            raise ValueError("When --use_parts is not set, --input_json is required.")

        with open(args.input_json, "r", encoding="utf-8") as f:
            entries = json.load(f)

        entry_ids = split_list(list(range(len(entries))), args.N, args.k)
        json_stem = os.path.splitext(os.path.basename(args.input_json))[0]
        save_base = os.path.join(args.save_root, json_stem)
        safe_makedirs(save_base)

        for idx in entry_ids:
            entry = entries[idx]

            if "video" in entry and entry["video"]:
                vp = entry["video"]
                video_path = vp if os.path.isabs(vp) else os.path.join(args.video_root, vp)
                stem = os.path.splitext(os.path.basename(video_path))[0]
            else:
                vid = entry.get("vid", None)
                if vid is None:
                    continue
                video_path = os.path.join(args.video_root, f"{vid}.mp4")
                stem = str(vid)

            if not os.path.exists(video_path):
                continue

            out_dir = os.path.join(save_base, stem)
            tasks.append({
                "video_path": video_path,
                "out_dir": out_dir,
                "stem": stem,
            })

    print(f"[rank {args.k}/{args.N}] Number of tasks: {len(tasks)}")

    # ------------------------------------------------------------
    # Execute tasks
    # ------------------------------------------------------------
    for t in tqdm(tasks, desc=f"rank {args.k}/{args.N}"):
        video_path = t["video_path"]
        out_dir = t["out_dir"]
        stem = t["stem"]

        safe_makedirs(out_dir)

        track_path = os.path.join(out_dir, "track.npy")
        vis_path = os.path.join(out_dir, "track_vis.npy")
        meta_path = os.path.join(out_dir, "meta.json")

        if args.skip_if_exist and os.path.exists(track_path) and os.path.exists(vis_path):
            continue

        try:
            # Always read full video first to get true frame count
            full_frames = read_video_local(video_path, max_frames=None, stride=1)
            full_T = len(full_frames)

            # Prevent accidental mismatch unless explicitly allowed
            if not args.allow_mismatch:
                if args.stride != 1:
                    raise ValueError("stride != 1 will cause frame mismatch. Use --allow_mismatch to override.")
                if args.max_frames is not None:
                    raise ValueError("max_frames is set and will cause frame mismatch. Use --allow_mismatch to override.")

            # Frames actually fed into CoTracker
            if args.stride == 1 and args.max_frames is None:
                frames = full_frames
            else:
                frames = read_video_local(video_path, max_frames=args.max_frames, stride=args.stride)

            video = build_video_tensor(frames, device=device)

            pred_tracks, pred_visibility = cotracker(video, grid_size=args.grid_size)

            # Shapes: (1, T, N, 2), (1, T, N) or (1, T, N, 1)
            pred_tracks = pred_tracks[0].cpu().numpy()
            pred_visibility = pred_visibility[0].cpu().numpy()

            if pred_visibility.ndim == 3 and pred_visibility.shape[-1] == 1:
                pred_visibility = pred_visibility[..., 0]

            track_T = pred_tracks.shape[0]

            if track_T != full_T and not args.allow_mismatch:
                raise ValueError(
                    f"Track length {track_T} does not match video frame count {full_T}."
                )
        
            # stx()

            np.save(track_path, pred_tracks)
            np.save(vis_path, pred_visibility)

            meta = {
                "video_path": video_path,
                "model_name": args.model_name,
                "grid_size": args.grid_size,
                "video_frame_count": full_T,
                "used_stride": args.stride,
                "used_max_frames": args.max_frames,
                "track_shape": list(pred_tracks.shape),
                "visibility_shape": list(pred_visibility.shape),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            print("===== ERROR =====")
            print("Video:", video_path)
            print("Output dir:", out_dir)
            print("Error:", repr(e))


if __name__ == "__main__":
    main()