# # depth -- original -- 已启动
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --test_dir data/examples/omnivcus/depth/ \
#   --num_frames 49 \
#   --out_dir results/VACE/depth/ \
#   --seed 0


# # mask -- original
# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel_all.py \
#   --test_dir data/examples/omnivcus/mask/ \
#   --num_frames 49 \
#   --out_dir results/VACE/mask/ \
#   --seed 0



# depth 1.3B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/VACE/depth/1.3B/ \
  --seed 0

# mask 1.3B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel_all.py \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/VACE/mask/1.3B/ \
  --seed 0


# depth 14B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --test_dir data/examples/omnivcus/depth/ \
  --num_frames 49 \
  --out_dir results/VACE/depth/14B/ \
  --seed 0

# mask 14B
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel_all.py \
  --test_dir data/examples/omnivcus/mask/ \
  --num_frames 49 \
  --out_dir results/VACE/mask/14B/ \
  --seed 0