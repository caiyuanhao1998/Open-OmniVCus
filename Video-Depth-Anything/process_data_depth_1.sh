#!/bin/bash

VIDEO_ROOT=../VideoCus-Factory/T2V_data/VIDGEN-1M_unzip
SAVE_ROOT=../VideoCus-Factory/T2V_data/label
JSON_PREFIX=../VideoCus-Factory/T2V_data/vidgen_filter_gemini_part_0_dpo

N=8   # 总卡数

for DPO_ID in $(seq 1 4); do
    INPUT_JSON=${JSON_PREFIX}_${DPO_ID}.json
    echo "Processing ${INPUT_JSON}"

    for ((G=0; G<8; G++)); do
        k=$((G+1))
        python process_data_depth.py \
            -G ${G} \
            -N ${N} \
            -k ${k} \
            --input_json ${INPUT_JSON} \
            --video_root ${VIDEO_ROOT} \
            --save_root ${SAVE_ROOT} &
    done

    wait
    echo "Finished ${INPUT_JSON}"
done