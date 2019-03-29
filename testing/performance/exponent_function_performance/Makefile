# My best guess
all:
	xlC_r -qsmp=omp -qoffload -c saxpy.cc -o saxpy_omp45_c.o
	xlcuf -c saxpy.cuf -o saxpy_cuda_f.o
	nvcc -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy.cu -o saxpy_cuda_c.o
	xlC_r -qsmp=omp -DOMP45 -DCUDA -DCUDAFORTRAN -c main.cc
	xlC_r -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_cuda_c.o -lxlf90_r -lxlcuf

clean:
	rm -f *.o mixed_lang_xlcuf mixed_lang_nvcc
	rm -f *.mod
	rm -f test_kernels
