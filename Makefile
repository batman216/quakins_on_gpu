EXE = quakins
CPP = nvc++

CPPFLAG = -std=c++20
GPUFLAG = -cudalib=cufft -lcufft  -I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/include

${EXE}: main_2d.cu
	${CPP} ${CPPFLAG} ${GPUFLAG} $^ -o $@
clean:
	rm quakins -f
