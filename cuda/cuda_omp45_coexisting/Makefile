# My best guess
all:
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -qoffload -c daxpy.cc -o daxpy_omp45_c.o
	xlcuf -O3 -c saxpy.cuf -o saxpy_cuda_f.o
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy.cu -o saxpy_cuda_c.o
	xlC_r -O3 -I$(CUDA_HOME)/include -qsmp=omp -c main.cc
	xlC_r -v -qtgtarch=sm_70 -W@,-v -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o daxpy_omp45_c.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib/ -lxlf90_r -lxlcuf -lxlfmath

debug:
	xlC_r -qcheck -O0 -g -qsmp=noopt -qoffload -c saxpy.cc -o daxpy_omp45_c.o
	xlcuf -qcheck -O0 -g -c saxpy.cuf -o saxpy_cuda_f.o
	nvcc -O0 -g -rdc=true -DNVCC -arch sm_70 --compiler-bindir xlC_r -c saxpy.cu -o saxpy_cuda_c.o
	xlC_r -qcheck -O0 -g -qsmp=noopt -c main.cc
	xlC_r -v -qcheck -O0 -g -qoffload -qsmp=noopt -qcuda -o test_kernels main.o saxpy_cuda_f.o daxpy_omp45_c.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o mixed_lang_xlcuf mixed_lang_nvcc
	rm -f *.mod
	rm -f test_kernels
