FULLDEBUG=-g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0

# Bug in XLF, workaround is to add '-d' to leave intermediate source file ( after preprocessing )

# OpenMP kernels
debugomp:
	xlC_r  $(FULLDEBUG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(FULLDEBUG) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -DOMP45C -DOMP45F -O0 -g $(FULLDEBUG)  -c main.cc
	xlC_r $(FULLDEBUG) -o test_kernels_debug main.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

# Both C and FORTRAN CUDA and OMP45 kernels, debug
debugall:
	xlC_r $(FULLDEBUG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(FULLDEBUG) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlf_r -d $(FULLDEBUG) -qextname=testsaxpy_cudafortran -c saxpy_cuda_f.F90 -o saxpy_cuda_f.o
	xlC_r -DCUDAF -DOMP45C -DOMP45F -O0 -g $(FULLDEBUG)  -c main.cc
	xlC_r $(FULLDEBUG) -o test_kernels_debug main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o  *.f90
	rm -f *.mod
	rm -f test_kernels_*
