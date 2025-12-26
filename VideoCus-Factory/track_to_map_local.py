import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import shutil
import random
import json

import numpy as np
import torch
import cv2
import imageio

from torchvision.utils import flow_to_image
from decord import VideoReader, cpu

import imageio.v3 as iio

from pdb import set_trace as stx


# =========================
# Gaussian blur utils
# =========================
def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack(
        (xx.reshape((kernel_size * kernel_size, 1)), yy.reshape((kernel_size * kernel_size, 1)))
    ).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


# =========================
# Flow building / visualization
# =========================
def get_cropped_flows(
    trajs: torch.Tensor,
    blur_kernel: np.ndarray,
    keep_min_prob: float,
    keep_max_prob: float,
    n_frames_out: int,
    out_w: int,
    out_h: int,
) -> torch.Tensor:
    """
    trajs: (B, N, T, 2) in pixel coords
    returns: (C, n_frames_out, H, W) in [-1,1]

    We always generate exactly n_frames_out frames for output.
    Frames beyond available track length will be all-zero flow.
    """
    T = int(trajs.shape[2])
    T_use = min(T, int(n_frames_out))

    optical_flow = np.zeros((n_frames_out, out_h, out_w, 2), dtype=np.float32)

    keep_prob = random.uniform(keep_min_prob, keep_max_prob)
    keep_mask = torch.rand(trajs.shape[1]) < keep_prob  # (N,)

    for traj_idx in range(trajs.shape[1]):
        if not keep_mask[traj_idx]:
            continue

        for frame_idx in range(T_use - 1):
            p = trajs[0, traj_idx, frame_idx]
            p1 = trajs[0, traj_idx, frame_idx + 1]

            x, y = float(p[0]), float(p[1])
            if x < 0 or x >= out_w or y < 0 or y >= out_h:
                continue

            # normalized flow
            optical_flow[frame_idx + 1, int(y), int(x), 0] = (float(p1[0]) - x) / out_w
            optical_flow[frame_idx + 1, int(y), int(x), 1] = (float(p1[1]) - y) / out_h

    for i in range(1, T_use):
        optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)

    optical_flow = torch.from_numpy(optical_flow)  # (T_out,H,W,2)

    flow_vis = optical_flow.permute(0, 3, 1, 2).clone()  # (T_out,2,H,W)
    flow_vis[:, 0] = flow_vis[:, 0] * out_w
    flow_vis[:, 1] = flow_vis[:, 1] * out_h

    rgb = flow_to_image(flow_vis)  # (T_out,3,H,W) uint8 [0,255]
    rgb = rgb.permute(1, 0, 2, 3).float() / 255.0 * 2 - 1  # (C,T_out,H,W) in [-1,1]
    return rgb


def write_rgb_video_uint8(rgb_thwc_uint8, save_path, fps):
    """
    rgb_thwc_uint8: (T,H,W,3) uint8
    """
    out_parent = os.path.dirname(save_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    with imageio.get_writer(save_path, mode="I", fps=float(fps)) as w:
        for fr in rgb_thwc_uint8:
            w.append_data(fr)


def read_video_local(video_path, max_frames=None, stride=1):
    """
    Read a local video using imageio.v3.

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


def get_src_avg_fps_decord(mp4_path):
    """
    Read average FPS from decord (most stable for mp4).
    """
    vr = VideoReader(mp4_path, ctx=cpu(0))
    return float(vr.get_avg_fps())


def load_tracks_npy(npy_path):
    """
    Supports:
      - (T, N, 2)
      - (N, T, 2)
    Returns: torch.Tensor (N, T, 2)
    """
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Unexpected npy shape {arr.shape}, expected (*,*,2).")

    # Heuristic: typically T >= N for tracking (e.g., 128 frames, 100 points)
    if arr.shape[0] >= arr.shape[1]:
        return torch.from_numpy(arr).permute(1, 0, 2).contiguous()  # (N,T,2)
    return torch.from_numpy(arr).contiguous()  # (N,T,2)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True, help="input track .npy (T,N,2 or N,T,2)")
    parser.add_argument("--mp4", type=str, required=True, help="input video .mp4")
    parser.add_argument("--out_dir", type=str, required=True, help="output folder")

    # If provided, override source fps; otherwise use source avg fps from decord.
    parser.add_argument("--fps", type=float, default=None, help="override output fps (default: use source avg fps)")
    parser.add_argument("--default_fps", type=float, default=24.0, help="fallback when source fps is unavailable")

    parser.add_argument("--keep_min_prob", type=float, default=0.0)
    parser.add_argument("--keep_max_prob", type=float, default=1.0)

    parser.add_argument("--blur_kernel_size", type=int, default=99)
    parser.add_argument("--blur_sigma", type=float, default=10.0)

    parser.add_argument("--name", type=str, default=None, help="output name prefix (default: derived from mp4 name)")
    parser.add_argument("--copy_video", action="store_true", help="also copy input mp4 into out_dir")

    parser.add_argument("--seed", type=int, default=None, help="random seed (optional)")

    parser.add_argument(
        "--use_track_len",
        action="store_true",
        help="If set, output flowvis length = track_T. Otherwise use video frame count.",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    base = os.path.splitext(os.path.basename(args.mp4))[0] if args.name is None else args.name

    # Read source fps (preferred) from decord
    try:
        src_avg_fps = get_src_avg_fps_decord(args.mp4)
    except Exception:
        src_avg_fps = None

    # Read frames using imageio (same as your track generation pipeline)
    video_frames = read_video_local(args.mp4)
    # stx()
    video_T = int(len(video_frames))
    h, w = video_frames[0].shape[:2]

    # Load tracks (N,T,2)
    tracks_nt2 = load_tracks_npy(args.npy)
    # stx()
    track_T = int(tracks_nt2.shape[1])

    # Decide output fps: force to match source video fps
    if args.fps is not None:
        out_fps = float(args.fps)
    else:
        out_fps = float(src_avg_fps) if src_avg_fps is not None else float(args.default_fps)

    # Decide output frame count
    if args.use_track_len:
        n_frames_out = track_T
    else:
        n_frames_out = video_T

    blur_kernel = bivariate_Gaussian(
        args.blur_kernel_size,
        args.blur_sigma,
        args.blur_sigma,
        0,
        grid=None,
        isotropic=True,
    ).astype(np.float32)

    flow_rgb_cthw = get_cropped_flows(
        trajs=tracks_nt2[None],  # (1,N,T,2)
        blur_kernel=blur_kernel,
        keep_min_prob=args.keep_min_prob,
        keep_max_prob=args.keep_max_prob,
        n_frames_out=n_frames_out,
        out_w=w,
        out_h=h,
    )  # (C,T_out,H,W) in [-1,1]

    flow_rgb_thwc = (
        ((flow_rgb_cthw.permute(1, 2, 3, 0) + 1) / 2 * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    vis_mp4 = os.path.join(args.out_dir, f"{base}_flowvis.mp4")
    write_rgb_video_uint8(flow_rgb_thwc, vis_mp4, fps=out_fps)

    copied_mp4 = None
    if args.copy_video:
        copied_mp4 = os.path.join(args.out_dir, f"{base}_input.mp4")
        shutil.copy2(args.mp4, copied_mp4)

    meta = {
        "input_npy": os.path.abspath(args.npy),
        "input_mp4": os.path.abspath(args.mp4),
        "out_dir": os.path.abspath(args.out_dir),
        "out_flow_vis_mp4": os.path.abspath(vis_mp4),
        "out_input_mp4": os.path.abspath(copied_mp4) if copied_mp4 else None,
        "src_n_frames_imageio": int(video_T),
        "src_avg_fps_decord": float(src_avg_fps) if src_avg_fps is not None else None,
        "used_out_fps": float(out_fps),
        "use_track_len": bool(args.use_track_len),
        "n_frames_out": int(n_frames_out),
        "height": int(h),
        "width": int(w),
        "tracks_shape_NT2": list(tracks_nt2.shape),
        "track_T": int(track_T),
        "keep_min_prob": float(args.keep_min_prob),
        "keep_max_prob": float(args.keep_max_prob),
        "blur_kernel_size": int(args.blur_kernel_size),
        "blur_sigma": float(args.blur_sigma),
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, f"{base}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" -", vis_mp4)
    if copied_mp4:
        print(" -", copied_mp4)
    print(
        f"Info: src_avg_fps={src_avg_fps}, used_out_fps={out_fps:.3f}, "
        f"video_T(imageio)={video_T}, track_T={track_T}, out_frames={n_frames_out}"
    )


if __name__ == "__main__":
    main()