# C version
copt:
	nvcc -O3 -lineinfo -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -O3 -g1 -qsmp=omp -qoffload -c main.cc -o main.o
	xlC_r -g1 -O3 -qoffload -qsmp=omp -qcuda -o test_kernel_c main.o saxpy_cuda_c.o

# FORTRAN version
fopt:
	xlf_r -d -qcuda -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	xlf_r -g1 -O3 -qcuda -qsmp=omp -qoffload -c main.f90
	xlf_r -g1 -O3 -qoffload -qsmp=omp -qcuda -o test_kernel_f main.o saxpy_cuda_f.o

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_kernel_*
