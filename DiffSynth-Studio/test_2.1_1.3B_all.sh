# depth - 1.3B - all
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/OmniVCus/Wan2.1-OmniVCus-1.3B/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/depth/1.3B/ \
  --seed 0

# mask - 1.3B - all
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/OmniVCus/Wan2.1-OmniVCus-1.3B/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/mask/1.3B/ \
  --seed 0