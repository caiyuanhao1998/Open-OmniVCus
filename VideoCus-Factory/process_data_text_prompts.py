import os
import json

json_path = 'T2V_data/vidgen_filter_gemini_part_0_dpo_0.json'
root_path = 'T2V_data/label/vidgen_filter_gemini_part_0_dpo_0'

# 读取 JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

assert isinstance(data, list), "JSON 顶层应为 list"

num_written = 0
num_skipped = 0

for item in data:
    # 安全检查
    if 'vid' not in item or 'original' not in item:
        num_skipped += 1
        continue

    vid = item['vid']
    prompt = item['original']

    # 对应的视频文件夹
    vid_dir = os.path.join(root_path, vid)
    os.makedirs(vid_dir, exist_ok=True)

    prompt_path = os.path.join(vid_dir, 'prompt.txt')

    # 写入 prompt.txt
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt.strip() + '\n')

    num_written += 1

print(f"✅ 写入完成: {num_written} 个 prompt.txt")
print(f"⚠️ 跳过无效项: {num_skipped} 个")