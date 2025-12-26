# depth -- full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/pre-trained/Wan2.2-OmniVCus-14B-high/epoch-0.safetensors \
  --ckpt_low  models/pre-trained/Wan2.2-OmniVCus-14B-low/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/depth/27_5/27.txt \
  --ref_video data/examples/omnivcus/depth/27_5/27.mp4 \
  --ref_image data/examples/omnivcus/depth/27_5/27.png \
  --num_frames 49 \
  --out results_video/OmniVCus/depth/14B_2.2/27.mp4 \
  --seed 0

# mask -- full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/pre-trained/Wan2.2-OmniVCus-14B-high/epoch-0.safetensors \
  --ckpt_low  models/pre-trained/Wan2.2-OmniVCus-14B-low/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/54/54.txt \
  --ref_video data/examples/omnivcus/mask/54/54.mp4 \
  --ref_image data/examples/omnivcus/mask/54/54.png \
  --num_frames 49 \
  --out results_video/OmniVCus/mask/14B_2.2/54.mp4 \
  --seed 0