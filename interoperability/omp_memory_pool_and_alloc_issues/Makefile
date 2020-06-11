xl:
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
	nvcc --compiler-bindir xlC_r --compiler-options -O3 --compiler-options -qsmp=omp -O3 -arch sm_70 -c daxpy.cu -o daxpy_cuda_c.o
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -c main.cc
	xlC_r -v -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o daxpy_omp45_c.o daxpy_cuda_c.o

# Not working, as yet.  Need to research how to link this correctly.
clang:
	nvcc --compiler-bindir clang++ -O3 -I$(CUDA_HOME)/include --compiler-options -fopenmp --compiler-options -fopenmp-targets=nvptx64-nvidia-cuda -c daxpy.cc -o daxpy_omp45_c.o
	nvcc --compiler-bindir clang++ -O3 --compiler-options -fopenmp -arch sm_70 -c daxpy.cu -o daxpy_cuda_c.o
	nvcc --compiler-bindir clang++ -O3 -I$(CUDA_HOME)/include --compiler-options -fopenmp -c main.cc
	nvcc --compiler-bindir clang++ -v -O3 --compiler-options -fopenmp --compiler-options -fopenmp-targets=nvptx64-nvidia-cuda -o test_kernels main.o daxpy_omp45_c.o daxpy_cuda_c.o

clean:
	rm -f *.mod
	rm -f *.o
	rm -f test_kernels
