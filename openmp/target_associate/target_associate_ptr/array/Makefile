include ../../../makefile_tools/xl.mk

opt:
	xlC_r -O3 -g1 -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -g1 -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -g1 -DOMP45C -DOMP45F -O3 -qsmp=omp -c main.cc
	xlC_r -g1 -O3 -qoffload -qsmp=omp -o test_kernels_opt main.o saxpy_omp45_c.o saxpy_omp45_f.o -L${XL_COMPILER_LIBS} -lxlf90_r -lxlcuf -lxlfmath -lxlcuf -lxlcufdevice 

clean:
	rm -f *.o F*.f90
	rm -f *.mod
	rm -f test_kernels_*
