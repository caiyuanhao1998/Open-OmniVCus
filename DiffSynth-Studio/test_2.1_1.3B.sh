# Depth  -----  1.3B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --ckpt models/pre-trained/Wan2.1-OmniVCus-1.3B/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/depth/26_2/26.txt \
  --ref_video data/examples/omnivcus/depth/26_2/26.mp4 \
  --ref_image data/examples/omnivcus/depth/26_2/26.png \
  --num_frames 49 \
  --out results_video/OmniVCus/depth/1.3B/26.mp4 \
  --seed 0

# Mask  -----  1.3B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --ckpt models/pre-trained/Wan2.1-OmniVCus-1.3B/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/32/32.txt \
  --ref_video data/examples/omnivcus/mask/32/32.mp4 \
  --ref_image data/examples/omnivcus/mask/32/32.png \
  --num_frames 49 \
  --out results_video/OmniVCus/mask/1.3B/32.mp4 \
  --seed 0