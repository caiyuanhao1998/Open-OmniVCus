# # depth -- only depth
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
#   --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/depth/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/depth/2.2_14B_only_depth/ \
#   --seed 0



# # depth -- only depth 1e-6 -- 已跑
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_depth_1e-6/epoch-0.safetensors \
#   --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_depth_1e-6/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/depth/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/depth/2.2_14B_only_depth_1e-6/ \
#   --seed 0


# # depth -- all 1e-6 -- 已跑
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_1e-6/epoch-0.safetensors \
#   --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_1e-6/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/depth/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/depth/2.2_14B_1e-6/ \
#   --seed 0


# # mask -- all_data 1e-6 -- 已跑
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_1e-6/epoch-0.safetensors \
#   --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_1e-6/epoch-0.safetensors \
#   --test_dir data/examples/omnivcus/mask/ \
#   --num_frames 49 \
#   --out_dir results/OmniVCus/mask/2.2_14B_1e-6/ \
#   --seed 0





# depth -- 1.3B -- only depth
# Depth  -----  1.3B only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all_depth/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/1.3B_only_depth/ \
  --seed 0



# depth -- 1.3B -- all data
# Depth  -----  1.3B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/depth/1.3B/ \
  --seed 0


# depth - 1.3B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/depth/1.3B/ \
  --seed 0



# depth -- 14B -- only depth
# Depth  -----  14B only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all_depth/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/depth/14B_only_depth/ \
  --seed 0

# depth -- 14B -- all data
# Depth  -----  14B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/depth/14B/ \
  --seed 0


# depth - 14B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/depth/14B/ \
  --seed 0




torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all_reverse.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/depth/14B/ \
  --seed 0


# mask -- all_data -- 1.3B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/mask/1.3B/ \
  --seed 0


# mask - 1.3B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/mask/1.3B/ \
  --seed 0


# mask -- all_data -- 14B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus/mask/14B/ \
  --seed 0


# mask - 14B - all parts
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_part_all_mix/epoch-0.safetensors \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/OmniVCus_all_parts/mask/14B/ \
  --seed 0
