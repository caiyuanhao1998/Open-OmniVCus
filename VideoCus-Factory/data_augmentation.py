#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import random
from typing import Optional, Tuple, List
from pdb import set_trace as stx

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def is_image_file(p: str) -> bool:
    return p.lower().endswith(IMG_EXTS)


def list_images(root: str) -> List[str]:
    if not root:
        return []
    if not os.path.isdir(root):
        return []
    out = []
    for fn in os.listdir(root):
        fp = os.path.join(root, fn)
        if os.path.isfile(fp) and is_image_file(fp):
            out.append(fp)
    return out


def infer_entity_id_from_filename(path: str) -> int:
    bn = os.path.basename(path)
    stem, _ = os.path.splitext(bn)
    if stem.startswith("entity_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except Exception:
                pass
    return -1


def bbox_from_binary_mask(bin_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 + 1, y1 + 1)


def random_crop_max_aspect(img: Image.Image, target_w: int, target_h: int, rng: random.Random) -> Image.Image:
    w, h = img.size
    target_ar = target_w / target_h
    src_ar = w / h

    if src_ar >= target_ar:
        crop_h = h
        crop_w = int(round(crop_h * target_ar))
    else:
        crop_w = w
        crop_h = int(round(crop_w / target_ar))

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    max_left = w - crop_w
    max_top = h - crop_h
    left = 0 if max_left <= 0 else rng.randint(0, max_left)
    top = 0 if max_top <= 0 else rng.randint(0, max_top)

    return img.crop((left, top, left + crop_w, top + crop_h))


def sample_background(
    W: int, H: int, rng: random.Random,
    bg_white_prob: float,
    bg_pool: List[str],
) -> Image.Image:
    # 先决定白 or 随机
    use_white = (rng.random() < bg_white_prob) or (len(bg_pool) == 0)
    if use_white:
        return Image.new("RGB", (W, H), (255, 255, 255))

    # 否则：随机背景（失败就回退白底）
    bg_path = rng.choice(bg_pool)
    try:
        bg = Image.open(bg_path).convert("RGB")
        bg = random_crop_max_aspect(bg, W, H, rng).resize((W, H), resample=Image.BICUBIC)
        return bg
    except Exception:
        return Image.new("RGB", (W, H), (255, 255, 255))


def apply_color_exposure_jitter(
    img: Image.Image,
    rng: random.Random,
    jitter_prob: float = 0.9,
    brightness: Tuple[float, float] = (0.90, 1.10),
    contrast: Tuple[float, float] = (0.90, 1.10),
    saturation: Tuple[float, float] = (0.90, 1.10),
    gamma: Tuple[float, float] = (0.95, 1.05),
    temperature: Tuple[float, float] = (0.98, 1.02),
) -> Image.Image:
    if rng.random() > jitter_prob:
        return img

    b = rng.uniform(*brightness)
    img = ImageEnhance.Brightness(img).enhance(b)

    c = rng.uniform(*contrast)
    img = ImageEnhance.Contrast(img).enhance(c)

    s = rng.uniform(*saturation)
    img = ImageEnhance.Color(img).enhance(s)

    g = rng.uniform(*gamma)
    if abs(g - 1.0) > 1e-3:
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0) ** g
        img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    t = rng.uniform(*temperature)
    if abs(t - 1.0) > 1e-3:
        arr = np.asarray(img).astype(np.float32)
        arr[..., 0] *= t
        arr[..., 2] /= max(t, 1e-6)
        arr = np.clip(arr, 0.0, 255.0)
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    return img


def augment_reference_in_folder(
    ref_path: str,
    mask_name: str = "mask.png",
    orig_name: str = "caption_frame.png",
    scale_range: Tuple[float, float] = (0.6, 1.4),
    rotate_deg: float = 25.0,
    bg_white_prob: float = 0.15,
    bg_rand_prob: float = 0.35,
    bg_pool: Optional[List[str]] = None,
    seed: Optional[int] = None,
    jitter_prob: float = 0.9,
    brightness: Tuple[float, float] = (0.90, 1.10),
    contrast: Tuple[float, float] = (0.90, 1.10),
    saturation: Tuple[float, float] = (0.90, 1.10),
    gamma: Tuple[float, float] = (0.95, 1.05),
    temperature: Tuple[float, float] = (0.98, 1.02),
) -> Optional[Image.Image]:
    rng = random.Random(seed) if seed is not None else random

    folder = os.path.dirname(ref_path)
    entity_id = infer_entity_id_from_filename(ref_path)
    if entity_id < 0:
        return None

    mask_path = os.path.join(folder, mask_name)
    orig_path = os.path.join(folder, orig_name)
    if (not os.path.exists(mask_path)) or (not os.path.exists(orig_path)):
        return None

    orig = Image.open(orig_path).convert("RGB")
    maskL = Image.open(mask_path).convert("L")
    W, H = orig.size

    mask_np = np.array(maskL, dtype=np.int32)
    bin_np = (mask_np == entity_id).astype(np.uint8) * 255
    bbox = bbox_from_binary_mask(bin_np)
    if bbox is None:
        return None

    crop_rgb = orig.crop(bbox)
    crop_m = Image.fromarray(bin_np, mode="L").crop(bbox)

    smin, smax = scale_range
    scale = rng.uniform(smin, smax)
    cw, ch = crop_rgb.size
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    crop_rgb = crop_rgb.resize((new_w, new_h), resample=Image.BICUBIC)
    crop_m = crop_m.resize((new_w, new_h), resample=Image.NEAREST)

    deg = rng.uniform(-rotate_deg, rotate_deg)
    crop_rgb = crop_rgb.rotate(deg, resample=Image.BICUBIC, expand=True)
    crop_m = crop_m.rotate(deg, resample=Image.NEAREST, expand=True)

    crop_rgb = apply_color_exposure_jitter(
        crop_rgb, rng,
        jitter_prob=jitter_prob,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        temperature=temperature,
    )

    bg_pool = bg_pool or []
    bg = sample_background(W, H, rng, bg_white_prob, bg_pool)

    pw, ph = crop_rgb.size
    if pw > W or ph > H:
        ratio = min(W / pw, H / ph)
        pw2 = max(1, int(round(pw * ratio)))
        ph2 = max(1, int(round(ph * ratio)))
        crop_rgb = crop_rgb.resize((pw2, ph2), resample=Image.BICUBIC)
        crop_m = crop_m.resize((pw2, ph2), resample=Image.NEAREST)
        pw, ph = pw2, ph2

    left = (W - pw) // 2
    top = (H - ph) // 2
    out = bg.copy()
    out.paste(crop_rgb, (left, top), mask=crop_m)
    return out


def make_aug_ref_path(ref_abs: str) -> str:
    folder = os.path.dirname(ref_abs)
    bn = os.path.basename(ref_abs)
    stem, _ = os.path.splitext(bn)
    return os.path.join(folder, f"aug_{stem}.png")


def csv_out_paths(csv_path: str):
    base, ext = os.path.splitext(csv_path)
    aug_csv = f"{base}_aug{ext}"
    depth_csv = f"{base}_aug_depth{ext}"
    mask_csv = f"{base}_aug_mask{ext}"
    return aug_csv, depth_csv, mask_csv


def is_depth_video(path_str: str) -> bool:
    return os.path.basename(path_str) == "depth.mp4"


def resolve_path(p: str, rel_base_abs: str, csv_dir_abs: str) -> str:
    """
    解析 CSV 里的路径：
    - 如果是绝对路径：直接用
    - 如果是相对路径：优先相对 rel_base，其次相对 csv 所在目录
    """
    if os.path.isabs(p):
        return p

    cand1 = os.path.abspath(os.path.join(rel_base_abs, p))
    if os.path.exists(cand1):
        return cand1

    cand2 = os.path.abspath(os.path.join(csv_dir_abs, p))
    return cand2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="train_csv/vidgen_filter_gemini_part_0_all.csv")
    ap.add_argument("--ref_col", type=str, default="vace_reference_image")
    ap.add_argument("--video_col", type=str, default="vace_video")

    ap.add_argument("--mask_name", type=str, default="mask.png")
    ap.add_argument("--orig_name", type=str, default="caption_frame.png")

    ap.add_argument("--scale_min", type=float, default=0.6)
    ap.add_argument("--scale_max", type=float, default=1.4)
    ap.add_argument("--rotate_deg", type=float, default=25.0)

    ap.add_argument("--bg_white_prob", type=float, default=0.15)
    ap.add_argument("--bg_rand_prob", type=float, default=0.35)
    ap.add_argument("--random_bg_dir", type=str, default="")

    ap.add_argument("--jitter_prob", type=float, default=0.9)
    ap.add_argument("--brightness_min", type=float, default=0.90)
    ap.add_argument("--brightness_max", type=float, default=1.10)
    ap.add_argument("--contrast_min", type=float, default=0.90)
    ap.add_argument("--contrast_max", type=float, default=1.10)
    ap.add_argument("--saturation_min", type=float, default=0.90)
    ap.add_argument("--saturation_max", type=float, default=1.10)
    ap.add_argument("--gamma_min", type=float, default=0.95)
    ap.add_argument("--gamma_max", type=float, default=1.05)
    ap.add_argument("--temp_min", type=float, default=0.98)
    ap.add_argument("--temp_max", type=float, default=1.02)

    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--max_rows", type=int, default=-1)
    ap.add_argument("--skip_if_exists", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug_first_n", type=int, default=0)

    # NEW: 你要写回 CSV 的“相对路径基准目录”
    ap.add_argument("--rel_base", type=str, default=".", help="Write CSV paths relative to this directory (default: current working dir).")

    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)

    rel_base_abs = os.path.abspath(args.rel_base)
    csv_dir_abs = os.path.dirname(os.path.abspath(args.csv_path))

    with open(args.csv_path, "r", encoding="utf-8") as f:
        total_rows = max(0, sum(1 for _ in f) - 1)
    if args.max_rows > 0:
        total_rows = min(total_rows, args.max_rows)

    bg_pool = list_images(args.random_bg_dir) if args.random_bg_dir else []
    if args.verbose:
        print(f"[INFO] random_bg_pool: {len(bg_pool)} images")
        print(f"[INFO] rel_base_abs: {rel_base_abs}")
        print(f"[INFO] csv_dir_abs : {csv_dir_abs}")

    aug_csv_path, depth_csv_path, mask_csv_path = csv_out_paths(args.csv_path)

    processed = 0
    saved = 0
    failed = 0
    skipped = 0

    with open(args.csv_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if args.ref_col not in reader.fieldnames:
            raise KeyError(f"Column '{args.ref_col}' not found. CSV columns: {reader.fieldnames}")
        if args.video_col not in reader.fieldnames:
            raise KeyError(f"Column '{args.video_col}' not found. CSV columns: {reader.fieldnames}")

        fieldnames = reader.fieldnames

        os.makedirs(os.path.dirname(os.path.abspath(aug_csv_path)) or ".", exist_ok=True)

        f_aug = open(aug_csv_path, "w", encoding="utf-8", newline="")
        f_depth = open(depth_csv_path, "w", encoding="utf-8", newline="")
        f_mask = open(mask_csv_path, "w", encoding="utf-8", newline="")

        try:
            w_aug = csv.DictWriter(f_aug, fieldnames=fieldnames)
            w_depth = csv.DictWriter(f_depth, fieldnames=fieldnames)
            w_mask = csv.DictWriter(f_mask, fieldnames=fieldnames)

            w_aug.writeheader()
            w_depth.writeheader()
            w_mask.writeheader()

            pbar = tqdm(reader, total=total_rows, desc="Augment ref & write CSV", dynamic_ncols=True)

            for i, row in enumerate(pbar):
                if args.max_rows > 0 and processed >= args.max_rows:
                    break
                processed += 1

                ref_path = row.get(args.ref_col, "")
                vid_path = row.get(args.video_col, "")

                if not ref_path:
                    failed += 1
                    pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
                    continue

                # resolve absolute ref path (support relative)
                ref_abs = resolve_path(ref_path, rel_base_abs, csv_dir_abs)
                if not os.path.exists(ref_abs):
                    failed += 1
                    if args.verbose:
                        tqdm.write(f"[WARN] ref not found: {ref_path} -> {ref_abs}")
                    pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
                    continue

                out_abs = make_aug_ref_path(ref_abs)  # absolute save path
                seed = (args.seed + i) if args.seed >= 0 else None

                if args.skip_if_exists and os.path.exists(out_abs):
                    skipped += 1
                else:
                    out_img = augment_reference_in_folder(
                        ref_abs,
                        mask_name=args.mask_name,
                        orig_name=args.orig_name,
                        scale_range=(args.scale_min, args.scale_max),
                        rotate_deg=args.rotate_deg,
                        bg_white_prob=args.bg_white_prob,
                        bg_rand_prob=args.bg_rand_prob,
                        bg_pool=bg_pool,
                        seed=seed,
                        jitter_prob=args.jitter_prob,
                        brightness=(args.brightness_min, args.brightness_max),
                        contrast=(args.contrast_min, args.contrast_max),
                        saturation=(args.saturation_min, args.saturation_max),
                        gamma=(args.gamma_min, args.gamma_max),
                        temperature=(args.temp_min, args.temp_max),
                    )
                    if out_img is None:
                        failed += 1
                        if args.verbose:
                            tqdm.write(f"[WARN] augment failed: {ref_abs}")
                        pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
                        continue

                    try:
                        out_img.save(out_abs)
                        saved += 1
                    except Exception as e:
                        failed += 1
                        if args.verbose:
                            tqdm.write(f"[WARN] save failed: {out_abs} err={e}")
                        pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
                        continue

                # write relative path to CSV
                out_rel = os.path.relpath(out_abs, start=rel_base_abs)

                new_row = dict(row)
                new_row[args.ref_col] = out_rel

                w_aug.writerow(new_row)
                if is_depth_video(vid_path):
                    w_depth.writerow(new_row)
                else:
                    w_mask.writerow(new_row)

                pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)

                if args.debug_first_n > 0 and processed >= args.debug_first_n:
                    tqdm.write("[DEBUG] Stop early due to --debug_first_n")
                    break

        finally:
            f_aug.close()
            f_depth.close()
            f_mask.close()

    print("[DONE]")
    print(f"  processed={processed} saved={saved} skipped={skipped} failed={failed}")
    print(f"  wrote: {aug_csv_path}")
    print(f"  wrote: {depth_csv_path}")
    print(f"  wrote: {mask_csv_path}")


if __name__ == "__main__":
    main()