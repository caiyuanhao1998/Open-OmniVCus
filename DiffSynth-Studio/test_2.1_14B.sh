# Depth  -----  14B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --ckpt models/pre-trained/Wan2.1-OmniVCus-14B/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/depth/33/33.txt \
  --ref_video data/examples/omnivcus/depth/33/33.mp4 \
  --ref_image data/examples/omnivcus/depth/33/33.png \
  --num_frames 49 \
  --out results_video/OmniVCus/depth/14B_2.1/33.mp4 \
  --seed 0


# Mask  -----  14B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --ckpt models/pre-trained/Wan2.1-OmniVCus-14B/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/65/65.txt \
  --ref_video data/examples/omnivcus/mask/65/65.mp4 \
  --ref_image data/examples/omnivcus/mask/65/65.png \
  --num_frames 49 \
  --out results_video/OmniVCus/mask/14B_2.1/65.mp4 \
  --seed 0