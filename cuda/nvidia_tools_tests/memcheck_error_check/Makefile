# My best guess
all:
	xlC_r -g -qsmp=omp -qoffload -qcuda -c saxpy.cc -o saxpy_omp45_c.o
	xlf_r -g -qsmp=noopt -qoffload -qcuda -O0 -c saxpy.cuf -o saxpy_cuda_f.o
	xlC_r -g -qsmp=omp -qoffload -qcuda -c main.cc
	xlC_r -g -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o
	rm -f *.mod
	rm -f test_kernels
