#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def iter_files_recursive(folder: Path):
    """Yield all files under folder recursively."""
    for p in folder.rglob("*"):
        if p.is_file():
            yield p


def build_copy_plan(root: Path, base_name: str, prefix: str, dpo_start: int, dpo_end: int):
    """
    Returns list of (src_file, dst_file).
    We copy files from each dpo_i/<subfolder>/** into base/<subfolder>/** (preserving relative path under subfolder).
    """
    base_dir = root / base_name
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    plan = []

    for i in range(dpo_start, dpo_end + 1):
        dpo_dir = root / f"{prefix}_dpo_{i}"
        if not dpo_dir.exists():
            # 如果缺某个 dpo 目录，直接跳过
            continue

        # 只处理 dpo_dir 的“一级子文件夹”
        for sub in sorted([p for p in dpo_dir.iterdir() if p.is_dir()]):
            dest_sub = base_dir / sub.name
            # 递归拿到该子文件夹下所有文件
            for src_file in iter_files_recursive(sub):
                rel = src_file.relative_to(sub)  # 相对 subfolder 的路径
                dst_file = dest_sub / rel
                plan.append((src_file, dst_file))

    return plan


def main():
    parser = argparse.ArgumentParser(
        description="Copy files from vidgen_filter_gemini_part_0_dpo_0..15 subfolders into vidgen_filter_gemini_part_0 same-named subfolders, with tqdm."
    )
    parser.add_argument("--root", type=str, default=".", help="包含这些 folder 的父目录（默认当前目录）")
    parser.add_argument("--base", type=str, default="vidgen_filter_gemini_part_0", help="目标 base 文件夹名")
    parser.add_argument("--prefix", type=str, default="vidgen_filter_gemini_part_0", help="dpo 文件夹前缀（默认 vidgen_filter_gemini_part_0）")
    parser.add_argument("--dpo_start", type=int, default=0, help="起始 dpo id（默认 0）")
    parser.add_argument("--dpo_end", type=int, default=15, help="结束 dpo id（默认 15）")
    parser.add_argument("--overwrite", action="store_true", help="若目标文件已存在则覆盖（默认跳过）")
    parser.add_argument("--dry_run", action="store_true", help="只打印计划，不实际复制")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base_dir = root / args.base

    plan = build_copy_plan(
        root=root,
        base_name=args.base,
        prefix=args.prefix,
        dpo_start=args.dpo_start,
        dpo_end=args.dpo_end,
    )

    if not plan:
        print("没有找到任何需要复制的文件（检查 root/base/prefix 是否正确）。")
        return

    skipped = 0
    copied = 0

    pbar = tqdm(plan, desc="Copying", unit="file")
    for src, dst in pbar:
        # 目标存在且不覆盖 -> skip
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            copied += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

        pbar.set_postfix(copied=copied, skipped=skipped)

    print(f"\nDone. base_dir = {base_dir}")
    print(f"Planned files: {len(plan)} | Copied: {copied} | Skipped: {skipped} | Overwrite: {args.overwrite} | Dry-run: {args.dry_run}")


if __name__ == "__main__":
    main()