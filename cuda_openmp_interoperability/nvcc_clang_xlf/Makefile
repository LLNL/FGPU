all: xlf_link xlc_link

xlf_link:
	clang++ -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	clang++ -O2 -fopenmp -c main.cc
	xlf_r -O2 -qoffload -qsmp=omp -qcuda -o test_xlf_link main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlfmath

xlc_link:
	clang++ -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	clang++ -O2 -fopenmp -c main.cc
	xlC_r -O2 -qoffload -qsmp=omp -o test_xlc_link main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlfmath

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_*
