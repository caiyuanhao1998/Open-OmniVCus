# #!/bin/bash

# VIDEO_ROOT=../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip
# SAVE_ROOT=../VideoCus-Factory/T2V_data/label
# JSON_PREFIX=../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_0_dpo

# N=8   # 总卡数

# for DPO_ID in $(seq 13 15); do
#     INPUT_JSON=${JSON_PREFIX}_${DPO_ID}.json
#     echo "Processing ${INPUT_JSON}"

#     for ((G=0; G<8; G++)); do
#         k=$((G+1))
#         python process_data_depth.py \
#             -G ${G} \
#             -N ${N} \
#             -k ${k} \
#             --input_json ${INPUT_JSON} \
#             --video_root ${VIDEO_ROOT} \
#             --save_root ${SAVE_ROOT} &
#     done

#     wait
#     echo "Finished ${INPUT_JSON}"
# done

# part_4
python process_data_depth.py -G 0 -N 8 -k 1 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 1 -N 8 -k 2 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 2 -N 8 -k 3 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 3 -N 8 -k 4 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 4 -N 8 -k 5 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 5 -N 8 -k 6 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 6 -N 8 -k 7 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
python process_data_depth.py -G 7 -N 8 -k 8 --input_json ../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_4.json --video_root ../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip --save_root ../VideoCus-Factory/T2V_data/label &
wait