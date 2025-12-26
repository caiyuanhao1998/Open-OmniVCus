import os
import glob
import csv
import json
from tqdm import tqdm
from pdb import set_trace as stx

out_dir = "train_csv"

suffix_list = ["aug", "aug_mask", "aug_depth"]

for suffix in suffix_list:

    merged_csv = os.path.join(out_dir, f"vidgen_filter_gemini_part_0_all_{suffix}.csv")

    all_rows = []
    header = None

    for dpo_id in range(16):
        csv_path = os.path.join(out_dir, f"vidgen_filter_gemini_part_0_dpo_{dpo_id}_{suffix}.csv")

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