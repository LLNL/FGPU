include ../../../makefile_tools/xl.mk

opt:
	xlf_r -g1 -qinfo=omperrtrace -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -g1 -O3 -qsmp=omp -c main.cc
	xlC_r -g1 -O3 -qoffload -qsmp=omp -o test_kernels_opt main.o saxpy_omp45_f.o -L${XL_COMPILER_LIBS} -lxlf90_r -lxlcuf -lxlfmath -lxlcuf -lxlcufdevice

clean:
	rm -f *.o F*.f90
	rm -f *.mod
	rm -f test_kernels_*
