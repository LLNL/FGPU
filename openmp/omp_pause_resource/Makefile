# My best guess
all:
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy.cu -o saxpy_cuda_c.o
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -c main.cc
	xlC_r -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o daxpy_omp45_c.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2020.06.25/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o mixed_lang_xlcuf mixed_lang_nvcc
	rm -f *.mod
	rm -f test_kernels
