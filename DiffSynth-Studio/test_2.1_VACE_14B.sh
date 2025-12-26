
# Depth  -----  14B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/depth/33/33.txt \
  --ref_video data/examples/omnivcus/depth/33/33.mp4 \
  --ref_image data/examples/omnivcus/depth/33/33.png \
  --num_frames 49 \
  --out results_video/VACE/depth/14B_2.1/33.mp4 \
  --seed 0


# Mask  -----  14B original
torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B_tensor_parallel.py \
  --input_txt data/examples/omnivcus/mask/65/65.txt \
  --ref_video data/examples/omnivcus/mask/65/65.mp4 \
  --ref_image data/examples/omnivcus/mask/65/65.png \
  --num_frames 49 \
  --out results_video/VACE/mask/14B_2.1/65.mp4 \
  --seed 0