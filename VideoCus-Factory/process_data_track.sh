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
python process_data_track.py -G 0 -k 1 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 1 -k 2 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 2 -k 3 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 3 -k 4 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 4 -k 5 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 5 -k 6 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 6 -k 7 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_track.py -G 7 -k 8 -N 8 --input_json T2V_data/json_files/part_${PART_ID}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
wait