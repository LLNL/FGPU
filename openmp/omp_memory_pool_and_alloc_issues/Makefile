xl:
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -c main.cc
	xlC_r -v -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -o test_kernels main.o daxpy_omp45_c.o

clang:
	clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -c daxpy.cc -o daxpy_omp45_c.o
	clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -c main.cc
	clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o test_kernels main.o daxpy_omp45_c.o

clean:
	rm -f *.o
	rm -f test_kernels
