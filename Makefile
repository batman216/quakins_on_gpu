EXE = quakins
CPP = nvc++

CPPFLAG = -std=c++20
GPUFLAG = -cudalib=cufft -lcufft  

${EXE}: main_2d.cu
	${CPP} ${CPPFLAG} ${GPUFLAG} $^ -o $@
clean:
	rm quakins -f
