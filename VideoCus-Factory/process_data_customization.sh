#!/usr/bin/env bash

# =========================
# Argument parsing (only --part_id is supported)
# =========================
PART_ID=0

# If the first argument is "--part_id", override the default PART_ID
if [ "$1" = "--part_id" ]; then
  PART_ID="$2"
fi

# =========================
# Parallel processing (8 GPUs)
# =========================
python process_data_customization.py -G 0 -N 8 -k 1 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 1 -N 8 -k 2 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 2 -N 8 -k 3 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 3 -N 8 -k 4 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 4 -N 8 -k 5 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 5 -N 8 -k 6 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 6 -N 8 -k 7 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_customization.py -G 7 -N 8 -k 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
wait