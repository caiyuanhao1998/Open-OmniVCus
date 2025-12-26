accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload


accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-14B_full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload





####################################################################################################################
################################################## Train with OmniVCus #############################################
####################################################################################################################

# 1.3B model with OmniVCus dataset
# accelerate launch examples/wanvideo/model_training/train_omnivcus.py \
#   --dataset_base_path ../VideoCus-Factory \
#   --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all.csv \
#   --data_file_keys "video,vace_video,vace_reference_image" \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 10 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-4 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus" \
#   --trainable_models "vace" \
#   --extra_inputs "vace_video,vace_reference_image" \
#   --use_gradient_checkpointing_offload



# single-gpu debuging for OmniVCus
# torchrun --standalone --nproc_per_node=1 examples/wanvideo/model_training/train.py \
#   --dataset_base_path ../VideoCus-Factory \
#   --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all.csv \
#   --data_file_keys "video,vace_video,vace_reference_image" \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-4 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus" \
#   --trainable_models "vace" \
#   --extra_inputs "vace_video,vace_reference_image" \
#   --use_gradient_checkpointing_offload


# wan2.1-1.3B model with all data
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all_aug.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload


# wan2.1-1.3B model with depth data
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all_aug_depth.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus_all_depth" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload



# wan2.1-14B model with all data
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all_aug.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-14B_full_OmniVCus_all" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload


# wan2.1-14B model with depth data
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/train_csv/vidgen_filter_gemini_part_0_all_aug_depth.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-14B_full_OmniVCus_all_depth" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload