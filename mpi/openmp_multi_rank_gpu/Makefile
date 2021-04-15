all:
	mpicxx -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
	mpicxx -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c main.cc
	mpicxx -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o daxpy_omp45_c.o

clean:
	rm -f test_kernels *.o
