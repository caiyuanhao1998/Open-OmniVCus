import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as iio
import imageio


# =========================
# Utils
# =========================
def split_list(lst, N, K):
    """Split lst into N contiguous chunks, return K-th chunk (1-indexed)."""
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


def extract_vid(entry):
    """
    JSON entry -> vid (stem, no .mp4).
    Supports:
      - {"vid": "..."}
      - {"video": "..."} / {"video_path": "..."} / {"name": "..."} / {"filename": "..."} / {"file": "..."}
      - "xxx" (string entry)
      - "xxx.mp4" (string entry)
    """
    if isinstance(entry, str):
        return os.path.splitext(os.path.basename(entry))[0]

    if "vid" in entry:
        return str(entry["vid"])

    for key in ["video", "video_path", "name", "filename", "file"]:
        if key in entry:
            v = str(entry[key])
            return os.path.splitext(os.path.basename(v))[0]

    raise KeyError(f"Cannot find video id in entry keys: {list(entry.keys())}")


# =========================
# Video IO (imageio)
# =========================
def read_video_local(video_path, max_frames=None, stride=1):
    """Read local video -> list of RGB uint8 frames."""
    frames = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if stride > 1 and (i % stride != 0):
            continue
        frames.append(frame)  # RGB uint8
        if max_frames is not None and len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return frames


def get_video_fps(video_path, default=24.0):
    """
    Read fps from metadata. If missing/invalid -> fallback to default.
    NOTE: Some videos may return 0/None; handle that.
    """
    try:
        meta = iio.immeta(video_path)
        fps = meta.get("fps", None)
        if fps is None:
            return float(default)
        fps = float(fps)
        if fps <= 0:
            return float(default)
        return fps
    except Exception:
        return float(default)


def write_depth_video(depth_frames_u8, save_path, fps):
    """
    depth_frames_u8: list of (H,W) uint8 in [0,255]
    Save as mp4 using imageio.get_writer + append_data.
    Write 3-channel grayscale to avoid colorspace weirdness.
    """
    with imageio.get_writer(save_path, mode="I", fps=float(fps)) as writer:
        for d in depth_frames_u8:
            rgb = np.repeat(d[:, :, None], 3, axis=2)  # (H,W,3)
            writer.append_data(rgb.astype(np.uint8))


# =========================
# Depth normalization
# =========================
def normalize_depth_to_u8(depth):
    """Per-frame normalize depth map to uint8 [0,255]."""
    dmin = float(depth.min())
    dmax = float(depth.max())
    if (dmax - dmin) < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)
    d = (depth - dmin) / (dmax - dmin)
    d = (d * 255.0).clip(0, 255)
    return d.astype(np.uint8)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser("Local depth generation using VideoDepthAnything (json-driven)")

    # parallel split
    parser.add_argument("-G", type=int, required=True, help="GPU id for CUDA_VISIBLE_DEVICES")
    parser.add_argument("-k", type=int, required=True, help="k-th chunk (1-indexed)")
    parser.add_argument("-N", type=int, required=True, help="split into N chunks")

    # json-driven IO
    parser.add_argument("--input_json", type=str, required=True, help="JSON list containing video ids")
    parser.add_argument("--video_root", type=str, required=True, help="root directory where <vid>.mp4 lives")
    parser.add_argument("--save_root", type=str, required=True, help="output root directory")

    # depth model settings
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_res", type=int, default=1280)  # kept for compatibility
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitl"])

    # video processing
    parser.add_argument("--max_len", type=int, default=-1, help="max number of frames to read (-1 means no limit)")
    parser.add_argument("--stride", type=int, default=1, help="read every `stride` frames")

    # behavior
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing depth.mp4")
    parser.add_argument("--fps_fallback", type=float, default=24.0, help="fallback fps if metadata missing/invalid")
    parser.add_argument(
        "--assert_len_match",
        action="store_true",
        help="assert len(depths) == len(input frames) (useful to catch resampling)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    # load VideoDepthAnything
    from video_depth_anything.video_depth import VideoDepthAnything

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    ckpt_path = f"./checkpoints/video_depth_anything_{args.encoder}.pth"
    video_depth_anything.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    video_depth_anything = video_depth_anything.to(device).eval()

    # load entries
    with open(args.input_json, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("input_json must be a JSON list")

    print(f"Loaded {len(entries)} entries from {args.input_json}")

    # split by entry index
    entry_indexes = split_list(list(range(len(entries))), args.N, args.k)
    print(f"Worker chunk: k={args.k}/{args.N}, num_entries={len(entry_indexes)}")

    # output base: save_root / json_stem / <vid> / depth.mp4
    json_stem = os.path.splitext(os.path.basename(args.input_json))[0]
    save_base = os.path.join(args.save_root, json_stem)
    os.makedirs(save_base, exist_ok=True)
    print("save_base:", save_base)

    max_frames = None if args.max_len == -1 else int(args.max_len)

    for idx in tqdm(entry_indexes):
        try:
            entry = entries[idx]
            vid = extract_vid(entry)

            video_path = os.path.join(args.video_root, f"{vid}.mp4")
            if not os.path.exists(video_path):
                print(f"[Missing video] {video_path}")
                continue

            vid_dir = os.path.join(save_base, vid)
            os.makedirs(vid_dir, exist_ok=True)

            depth_save_path = os.path.join(vid_dir, "depth.mp4")
            if (not args.overwrite) and os.path.exists(depth_save_path):
                continue

            # fps = original video fps (single source of truth)
            video_fps = get_video_fps(video_path, default=args.fps_fallback)

            # read frames
            frames_np_lst = read_video_local(
                video_path,
                max_frames=max_frames,
                stride=int(args.stride),
            )
            frames = np.asarray(frames_np_lst, dtype=np.uint8)  # (T,H,W,3)

            # infer depth: target_fps MUST equal input fps (as requested)
            depths, _ = video_depth_anything.infer_video_depth(
                frames,
                target_fps=int(round(video_fps)),
                input_size=int(args.input_size),
                device=device,
            )

            if args.assert_len_match:
                if len(depths) != len(frames):
                    raise RuntimeError(
                        f"len(depths) != len(frames): {len(depths)} vs {len(frames)} "
                        f"(video_fps={video_fps}, stride={args.stride}, max_frames={max_frames})"
                    )

            depth_u8_list = [normalize_depth_to_u8(d) for d in depths]

            # write depth.mp4 with original fps
            write_depth_video(depth_u8_list, depth_save_path, fps=video_fps)

        except Exception as e:
            print("===== error ====")
            print("entry idx:", idx)
            try:
                print("vid:", extract_vid(entries[idx]))
            except Exception:
                pass
            print("err:", repr(e))


if __name__ == "__main__":
    main()