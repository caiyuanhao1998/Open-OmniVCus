import os
import glob
import csv
import json
from tqdm import tqdm
from pdb import set_trace as stx

out_dir = "T2V_data/train_csv"

for part_id in tqdm(range(8)):

    root_path = f"T2V_data/label/part_{part_id}"
    mp4_root = "T2V_data/VIDGEN-1M_unzip"
    out_csv = os.path.join(out_dir, f"part_{part_id}.csv")

    os.makedirs(out_dir, exist_ok=True)

    data_item = []

    fieldnames = ["video", "prompt", "vace_video", "vace_reference_image"]


    def read_text(path):
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()


    def read_prompt_any(d):
        """优先 prompt.txt，其次 caption_frame.json"""
        prompt_path = os.path.join(d, "prompt.txt")
        if os.path.exists(prompt_path):
            return read_text(prompt_path)

        json_path = os.path.join(d, "caption_frame.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                for k in ["prompt", "caption", "text", "description"]:
                    if k in obj and isinstance(obj[k], str):
                        return obj[k].strip()
            except Exception:
                pass

        return ""


    subdirs = sorted(
        d for d in glob.glob(os.path.join(root_path, "*"))
        if os.path.isdir(d)
    )


    for d in tqdm(subdirs):
        folder_name = os.path.basename(d)

        # 1) entity reference images
        ref_candidates = sorted(glob.glob(os.path.join(d, "entity*.png")))
        if not ref_candidates:
            continue

        # 2) prompt
        prompt = read_prompt_any(d)

        # 3) 原始 video（必须存在，否则直接跳过）
        video_path = os.path.join(mp4_root, f"{folder_name}.mp4")

        # stx()

        if not os.path.exists(video_path):
            print(f"{video_path} not found")
            continue  # ← 关键：video 不存在，整个样本跳过

        # 4) vace videos
        depth_path = os.path.join(d, "depth.mp4")
        has_depth = os.path.exists(depth_path)

        for ref_img in ref_candidates:
            vace_reference_image = ref_img
            base = os.path.splitext(os.path.basename(ref_img))[0]  # entity_20
            entity_mp4 = os.path.join(d, f"{base}.mp4")

            # (A) depth.mp4
            if has_depth:
                data_item.append({
                    "video": video_path,
                    "prompt": prompt,
                    "vace_video": depth_path,
                    "vace_reference_image": vace_reference_image,
                })

            # (B) entity_XX.mp4
            if os.path.exists(entity_mp4):
                data_item.append({
                    "video": video_path,
                    "prompt": prompt,
                    "vace_video": entity_mp4,
                    "vace_reference_image": vace_reference_image,
                })

    # 写 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_item)

    print(f"Saved {len(data_item)} rows to {out_csv}")



# ===========================================================
# Merge 8 CSV files
# ===========================================================

merged_csv = os.path.join(out_dir, "part_all.csv")

all_rows = []
header = None

for part_id in range(8):
    csv_path = os.path.join(out_dir, f"part_{part_id}.csv")

    if not os.path.exists(csv_path):
        print(f"[Skip] {csv_path} not found")
        continue

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 记录一次表头
        if header is None:
            header = reader.fieldnames

        for row in reader:
            all_rows.append(row)

# 写 merged CSV
with open(merged_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"[Done] Merged {len(all_rows)} rows into {merged_csv}")