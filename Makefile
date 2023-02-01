quakins: main_1d.cu
	nvc++ -o quakins -std=c++20 -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/include main_1d.cu
clean:
	rm quakins -f
