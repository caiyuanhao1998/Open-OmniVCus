##########################################################################################################################
#####################################################       1.3B Model     ################################################
##########################################################################################################################

# Depth  -----  1.3B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/VACE/video_Wan2.1-VACE-1.3B_8.mp4 \
  --seed 0

# Depth  -----  1.3B only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all_depth/epoch-0.safetensors \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/1.3B_only_depth/video_Wan2.1-VACE-1.3B_8.mp4 \
  --seed 0


# Depth  -----  1.3B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/1.3B/video_Wan2.1-VACE-1.3B_8.mp4 \
  --seed 0





# Mask  -----  1.3B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/VACE/mask/video_Wan2.1-VACE-1.3B_44.mp4 \
  --seed 0

# Mask  -----  1.3B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/OmniVCus/mask/1.3B/video_Wan2.1-VACE-1.3B_44.mp4 \
  --seed 0


##########################################################################################################################
#####################################################       14B Model     ################################################
##########################################################################################################################


# Depth  -----  14B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/VACE/video_Wan2.1-VACE-14B_8.mp4 \
  --seed 0

# Depth  -----  14B only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all_depth/epoch-0.safetensors \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/14B_only_depth/video_Wan2.1-VACE-14B_8.mp4 \
  --seed 0


# Depth  -----  14B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivucs/8/8.txt \
  --ref_video data/examples/omnivucs/8/8.mp4 \
  --ref_image data/examples/omnivucs/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/14B/video_Wan2.1-VACE-14B_8.mp4 \
  --seed 0



# Mask  -----  14B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/VACE/mask/video_Wan2.1-VACE-14B_44.mp4 \
  --seed 0



# Mask  -----  14B full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --ckpt models/train/Wan2.1-VACE-14B_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/OmniVCus/mask/14B/video_Wan2.1-VACE-14B_44.mp4 \
  --seed 0