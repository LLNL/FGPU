opt:
	clang++ -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -v --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2 -rdc=true -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	clang++ -O2 -fopenmp -c main.cc
	xlf_r -O2 -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_kernels
