FULLDEBUG=-g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0

# Both CUDA and OMP45 kernel, optimized
opt:
	CC -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70 -O3 -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	ftn -hlist=a -homp -O3 -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	CC -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70 -O3 -c main.cc -o main.o
#	CC -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70 -O3 -o test_kernels_opt main.o saxpy_omp45_c.o  -L${CUDA_HOME}/lib64 -lnvToolsExt -lcraymp
	CC -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70 -O3 -o test_kernels_opt main.o saxpy_omp45_c.o saxpy_omp45_f.o -L${CUDA_HOME}/lib64 -lnvToolsExt -lcraymp
#	xlf_r -g1 -qinfo=omperrtrace -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
#	xlC_r -I$(CUDA_HOME)/include -O3 -g1 -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
#	xlC_r -g1 -O3 -qsmp=omp -c main.cc
#	xlC_r -g1 -O3 -qoffload -qsmp=omp -o test_kernels_opt main.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath -lxlcuf -lxlcufdevice -L$(CUDA_HOME)/lib64 -lnvToolsExt

clean:
	rm -f *.o F*.f90
	rm -f *.mod
	rm -f test_kernels_*

# Both CUDA and OMP45 kernel, optimized
cpu:
	xlC_r -O3 -g1 -qsmp=omp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -g1 -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -g1 -DOMP45C -DOMP45F -O3 -qsmp=omp -c main.cc
	xlC_r -g1 -O3 -qsmp=omp -o test_kernels_opt main.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlfmath
