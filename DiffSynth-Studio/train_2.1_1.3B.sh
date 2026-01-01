####################################################################################################################
################################################## Train with OmniVCus #############################################
####################################################################################################################

# part_0
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/T2V_data/train_csv/part_0_aug.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus_part_0" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload

# # part_all
# accelerate launch examples/wanvideo/model_training/train.py \
#   --dataset_base_path ../VideoCus-Factory \
#   --dataset_metadata_path ../VideoCus-Factory/T2V_data/train_csv/part_all_aug.csv \
#   --data_file_keys "video,vace_video,vace_reference_image" \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus_part_all" \
#   --trainable_models "vace" \
#   --extra_inputs "vace_video,vace_reference_image" \
#   --use_gradient_checkpointing_offload



# part_all_mix
# accelerate launch examples/wanvideo/model_training/train.py \
#   --dataset_base_path ../VideoCus-Factory \
#   --dataset_metadata_path ../VideoCus-Factory/T2V_data/train_csv/part_all_aug_mix.csv \
#   --data_file_keys "video,vace_video,vace_reference_image" \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "./models/train/Wan2.1-VACE-1.3B_full_OmniVCus_part_all_mix" \
#   --trainable_models "vace" \
#   --extra_inputs "vace_video,vace_reference_image" \
#   --use_gradient_checkpointing_offload
