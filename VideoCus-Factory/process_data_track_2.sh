# source /sensei-fs/users/xic/opt/sam2/bin/activate

# 把 part_indexes 分成 N 份，取第 k 份
# python process_data_video_to_track.py -G 0 -k 1 -N 8 &
# python process_data_video_to_track.py -G 1 -k 2 -N 8 &
# python process_data_video_to_track.py -G 2 -k 3 -N 8 &
# python process_data_video_to_track.py -G 3 -k 4 -N 8 &
# python process_data_video_to_track.py -G 4 -k 5 -N 8 &
# python process_data_video_to_track.py -G 5 -k 6 -N 8 &
# python process_data_video_to_track.py -G 6 -k 7 -N 8 &
# python process_data_video_to_track.py -G 7 -k 8 -N 8 &
# wait
for PART in 2 3; do
    echo "===== Processing part ${PART} ====="
    python process_data_track.py -G 0 -k 1 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 1 -k 2 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 2 -k 3 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 3 -k 4 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 4 -k 5 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 5 -k 6 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 6 -k 7 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    python process_data_track.py -G 7 -k 8 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_${PART}.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
    wait
done