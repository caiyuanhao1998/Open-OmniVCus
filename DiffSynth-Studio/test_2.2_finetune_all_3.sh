# depth -- all_data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/depth/2.2_14B/ \
  --seed 0


# depth -- all parts - 14B 2.2
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/depth/2.2_14B/ \
  --seed 0


# mask -- all_data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/mask/2.2_14B/ \
  --seed 0


# mask -- all parts - 14B 2.2
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/mask/2.2_14B/ \
  --seed 0


# # mask -- all_data -- 1.3B
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
#   --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/mask/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/mask/1.3B/ \
#   --seed 0

# # mask -- all_data -- 14B
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
#   --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/mask/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/mask/14B/ \
#   --seed 0