import csv
import os

IN_CSV_1 = "T2V_data/train_csv/part_all_aug_depth.csv"
IN_CSV_2 = "T2V_data/train_csv/part_all_aug_mask.csv"
OUT_CSV  = "T2V_data/train_csv/part_all_aug_mix.csv"

def count_data_rows(path: str) -> int:
    """Count data rows (exclude header)."""
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            next(r)  # skip header
        except StopIteration:
            return 0
        return sum(1 for _ in r)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 统计原始数据行数（不含 header）
n1 = count_data_rows(IN_CSV_1)
n2 = count_data_rows(IN_CSV_2)

cnt1_sel = 0  # csv1 被选中（奇数行）
cnt2_sel = 0  # csv2 被选中（偶数行）
cnt_out  = 0  # 输出数据行数

with open(IN_CSV_1, "r", encoding="utf-8") as f1, \
     open(IN_CSV_2, "r", encoding="utf-8") as f2, \
     open(OUT_CSV,  "w", encoding="utf-8", newline="") as fout:

    r1 = csv.reader(f1)
    r2 = csv.reader(f2)
    w  = csv.writer(fout)

    h1 = next(r1)
    h2 = next(r2)
    assert h1 == h2, "两个 CSV 的 header 不一致！"
    w.writerow(h1)

    # i1/i2 是“当前读到的源文件数据行号”（从 1 开始，不含 header）
    i1 = 0
    i2 = 0

    # 输出的目标行号 i（从 1 开始，不含 header）
    i = 1
    while True:
        if i % 2 == 1:
            # 需要 csv1 的第 i 行（奇数行）
            try:
                while i1 < i:
                    row1 = next(r1)
                    i1 += 1
                # 这里保证 i1 == i
                w.writerow(row1)
                cnt1_sel += 1
                cnt_out += 1
            except StopIteration:
                break
        else:
            # 需要 csv2 的第 i 行（偶数行）
            try:
                while i2 < i:
                    row2 = next(r2)
                    i2 += 1
                # 这里保证 i2 == i
                w.writerow(row2)
                cnt2_sel += 1
                cnt_out += 1
            except StopIteration:
                break

        i += 1

print(f"CSV1 total data rows: {n1}")
print(f"CSV2 total data rows: {n2}")
print(f"CSV1 odd rows selected: {cnt1_sel}")
print(f"CSV2 even rows selected: {cnt2_sel}")
print(f"Output CSV data rows: {cnt_out}")
print(f"Saved mixed CSV to: {OUT_CSV}")