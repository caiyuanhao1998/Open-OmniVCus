##########################################################################################################################
#####################################################       14B Model     ################################################
##########################################################################################################################

# depth -- original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/8/8.txt \
  --ref_video data/examples/omnivcus/8/8.mp4 \
  --ref_image data/examples/omnivcus/8/8.png \
  --num_frames 49 \
  --out results/VACE/video_Wan2.2-VACE-14B_8.mp4 \
  --seed 0


# depth -- only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/8/8.txt \
  --ref_video data/examples/omnivcus/8/8.mp4 \
  --ref_image data/examples/omnivcus/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/2.2_14B_only_depth/video_Wan2.2-VACE-14B_8.mp4 \
  --seed 0


# depth -- full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/8/8.txt \
  --ref_video data/examples/omnivcus/8/8.mp4 \
  --ref_image data/examples/omnivcus/8/8.png \
  --num_frames 49 \
  --out results/OmniVCus/2.2_14B/video_Wan2.2-VACE-14B_8.mp4 \
  --seed 0



############################################ for flexible ###########################################

# depth -- original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/depth/22_3/22.txt \
  --ref_video data/examples/omnivcus/depth/22_3/22.mp4 \
  --ref_image data/examples/omnivcus/depth/22_3/22.png \
  --num_frames 49 \
  --out results/VACE/depth/video_Wan2.2-VACE-14B_22_3.mp4 \
  --seed 0


# depth -- only depth
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all_depth/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/depth/22_3/22.txt \
  --ref_video data/examples/omnivcus/depth/22_3/22.mp4 \
  --ref_image data/examples/omnivcus/depth/22_3/22.png \
  --num_frames 49 \
  --out results/OmniVCus/depth/2.2_14B_only_depth/video_Wan2.2-VACE-14B_22_3.mp4 \
  --seed 0


# depth -- full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/depth/22_3/22.txt \
  --ref_video data/examples/omnivcus/depth/22_3/22.mp4 \
  --ref_image data/examples/omnivcus/depth/22_3/22.png \
  --num_frames 49 \
  --out results/OmniVCus/depth/2.2_14B/video_Wan2.2-VACE-14B_22_3.mp4 \
  --seed 0







# mask -- original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/VACE/mask/video_Wan2.2-VACE-14B_44.mp4 \
  --seed 0


# mask -- full data
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B_tensor_parallel.py \
  --ckpt_high models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all/epoch-0.safetensors \
  --ckpt_low  models/train/Wan2.2-VACE-Fun-A14B_low_noise_full_OmniVCus_all/epoch-0.safetensors \
  --input_txt data/examples/omnivcus/mask/44/44.txt \
  --ref_video data/examples/omnivcus/mask/44/44.mp4 \
  --ref_image data/examples/omnivcus/mask/44/44.png \
  --num_frames 49 \
  --out results/OmniVCus/mask/2.2_14B/video_Wan2.2-VACE-14B_44.mp4 \
  --seed 0






##########################################################################################################################
###################################################     All Evaluate     #################################################
##########################################################################################################################