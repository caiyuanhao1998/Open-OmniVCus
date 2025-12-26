# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control_offload.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control_offload_dynamic.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control_offload_disk.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B.py

torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B_tensor_parallel.py