# depth - 14B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/OmniVCus/Wan2.1-OmniVCus-14B/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/depth/14B_2.1/ \
  --seed 0

# mask - 14B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/OmniVCus/Wan2.1-OmniVCus-14B/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/mask/14B_2.1/ \
  --seed 0