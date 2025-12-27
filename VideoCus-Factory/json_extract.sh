for part_idx in $(seq 0 7); do
    python json_extract.py --input T2V_data/vidgen_filter_gemini_part_${part_idx}.json --output T2V_data/part_${part_idx}.json
done