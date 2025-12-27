##################################################################################################################################################
#########################################################      Train OmniVCus      ###############################################################
##################################################################################################################################################

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ../VideoCus-Factory \
  --dataset_metadata_path ../VideoCus-Factory/T2V_data/train_csv/part_0_aug.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.2-VACE-Fun-A14B_high_noise_full_OmniVCus_all" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --initialize_model_on_cpu