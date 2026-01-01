import csv
import os
import argparse

FIELDNAMES = ["video", "prompt", "vace_video", "vace_reference_image"]


def to_aug_entity_png(p: str) -> str:
    """
    Replace the basename 'entity_*.png' -> 'aug_entity_*.png' in vace_reference_image.
    Only modifies when basename startswith 'entity_' and endswith '.png'.
    """
    if not p:
        return p
    d, base = os.path.split(p)
    if base.startswith("entity_") and base.endswith(".png"):
        base = "aug_" + base
        return os.path.join(d, base)
    return p


def is_depth_video(p: str) -> bool:
    return bool(p) and p.replace("\\", "/").endswith("/depth.mp4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split augmented CSV into aug / depth / mask parts"
    )
    parser.add_argument(
        "--part_id",
        type=str,
        default="0",
        choices=[str(i) for i in range(8)] + ["all"],
        help="Part id of CSV file: 0–7 or 'all' (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    part_id = args.part_id

    # =========================
    # Paths constructed from part_id
    # =========================
    if part_id == "all":
        IN_CSV = "T2V_data/train_csv/part_all.csv"
        OUT_AUG = "T2V_data/train_csv/part_all_aug.csv"
        OUT_DEPTH = "T2V_data/train_csv/part_all_aug_depth.csv"
        OUT_MASK = "T2V_data/train_csv/part_all_aug_mask.csv"
    else:
        IN_CSV = f"T2V_data/train_csv/part_{part_id}.csv"
        OUT_AUG = f"T2V_data/train_csv/part_{part_id}_aug.csv"
        OUT_DEPTH = f"T2V_data/train_csv/part_{part_id}_aug_depth.csv"
        OUT_MASK = f"T2V_data/train_csv/part_{part_id}_aug_mask.csv"

    with open(IN_CSV, "r", encoding="utf-8", newline="") as fin, \
         open(OUT_AUG, "w", encoding="utf-8", newline="") as f_aug, \
         open(OUT_DEPTH, "w", encoding="utf-8", newline="") as f_depth, \
         open(OUT_MASK, "w", encoding="utf-8", newline="") as f_mask:

        reader = csv.DictReader(fin)

        w_aug = csv.DictWriter(f_aug, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        w_depth = csv.DictWriter(f_depth, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        w_mask = csv.DictWriter(f_mask, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)

        w_aug.writeheader()
        w_depth.writeheader()
        w_mask.writeheader()

        n_total = n_depth = n_mask = 0

        for row in reader:
            out = {k: row.get(k, "") for k in FIELDNAMES}
            out["vace_reference_image"] = to_aug_entity_png(
                out["vace_reference_image"]
            )

            w_aug.writerow(out)
            n_total += 1

            if is_depth_video(out["vace_video"]):
                w_depth.writerow(out)
                n_depth += 1
            else:
                w_mask.writerow(out)
                n_mask += 1

    print(f"Done.\n  total={n_total}\n  depth={n_depth}\n  mask={n_mask}")
    print(f"Wrote:\n  {OUT_AUG}\n  {OUT_DEPTH}\n  {OUT_MASK}")


if __name__ == "__main__":
    main()