for dpo_idx in $(seq 8 9); do
  echo "===== Running dpo_idx=${dpo_idx} ====="
  python data_augmentation.py \
    --csv_path train_csv/vidgen_filter_gemini_part_0_dpo_${dpo_idx}.csv \
    --ref_col vace_reference_image \
    --video_col vace_video \
    --random_bg_dir T2I_data/data_1024_10K/data_1024_10K/data_000000 \
    --bg_white_prob 0.15 --bg_rand_prob 0.85 \
    --scale_min 0.6 --scale_max 1.4 --rotate_deg 25 \
    --jitter_prob 0.9 \
    --brightness_min 0.90 --brightness_max 1.10 \
    --contrast_min 0.90 --contrast_max 1.10 \
    --saturation_min 0.90 --saturation_max 1.10 \
    --gamma_min 0.95 --gamma_max 1.05 \
    --temp_min 0.98 --temp_max 1.02 \
    --verbose
done