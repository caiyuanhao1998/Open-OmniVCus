# depth -- all parts - 14B 2.2
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/OmniVCus/Wan2.2-OmniVCus-14B-high/epoch-0.safetensors \
  --ckpt_low  models/OmniVCus/Wan2.2-OmniVCus-14B-low/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/depth/2.2_14B/ \
  --seed 0


# mask -- all parts - 14B 2.2
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/OmniVCus/Wan2.2-OmniVCus-14B-high/epoch-0.safetensors \
  --ckpt_low  models/OmniVCus/Wan2.2-OmniVCus-14B-low/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results_video/OmniVCus/mask/2.2_14B/ \
  --seed 0