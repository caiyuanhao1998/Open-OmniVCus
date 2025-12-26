python process_data_depth.py -G 0 -k 1 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 1 -k 2 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 2 -k 3 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 3 -k 4 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 4 -k 5 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 5 -k 6 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 6 -k 7 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
python process_data_depth.py -G 7 -k 8 -N 8 --input_json T2V_data/vidgen_filter_gemini_part_1.json --video_root T2V_data/VIDGEN-1M_unzip --save_root T2V_data/label &
wait