# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-14B.py

torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-14B_tensor_parallel.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_tensor_parallel.py

torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_tensor_parallel_v2.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_sequence_parallel.py

# torchrun --standalone --nproc_per_node=8 examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py