FULLDEBUG=-g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0

# Both CUDA and OMP45 kernel, optimized
opt:
	xlC_r -O3 -g1 -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -g1 -O3 -qsmp=omp -qoffload -c openmp_tools.F90 -o openmp_tools.o
	xlf_r -g1 -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -g1 -O3 -qsmp=omp -c main.cc
	xlC_r -g1 -O3 -qoffload -qsmp=omp -o test_kernels_opt main.o saxpy_omp45_c.o openmp_tools.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-beta-2019.06.19/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath -lxlcuf -lxlcufdevice 
clean:
	rm -f *.o F*.f90
	rm -f *.mod
	rm -f test_kernels_*
