all:
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
#	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c daxpy.cu -o daxpy_cuda_c.o
	nvcc --compiler-bindir clang++ --compiler-options -O3 --compiler-options -fopenmp -O3  -arch sm_70 -c daxpy.cu -o daxpy_cuda_c.o
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -c main.cc
#	xlC_r -v -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o daxpy_omp45_c.o daxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib/ -lxlf90_r -lxlcuf -lxlfmath
	xlf_r -v -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o daxpy_omp45_c.o daxpy_cuda_c.o

clean:
	rm -f *.mod
	rm -f *.o
	rm -f test_kernels
