# cuda virtual memory benchmark 

## Update Makefile for benchmark
1. update Makefile and `make`
2. get profile of the cuda api `nsys profile ./host_example`
3. get cuda api from Nvidia Nsight system event view
4. csv file process with `python data_proc.py`

# REFERENCE

[API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA)

[CODE-EXAMPLE](https://github.com/NVIDIA-developer-blog/code-samples/tree/52b16fac9a135ca12b6c4d53529128d2672cc6ad/posts/cuda-vmm)
