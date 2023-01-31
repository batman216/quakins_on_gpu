quakins: main.cu
	nvc++ -o quakins -std=c++20 -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/include main.cu
clean:
	rm quakins -f
