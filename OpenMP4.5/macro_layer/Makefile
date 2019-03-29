FLINK = -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

xl_omp:
	xlf_r -DENABLE_OMP_MACRO -qsmp=omp -qoffload -o test_xl_omp saxpy_omp45_f.F90

xl:
	xlf_r -o test_xl_omp saxpy_omp45_f.F90

intel_omp:
	ifort -DENABLE_OMP_MACRO -allow nofpp-comments -o test_intel_omp saxpy_omp45_f.F90

intel:
	ifort -allow nofpp-comments -o test_intel saxpy_omp45_f.F90

gnu_omp:
	gfortran -DENABLE_OPENMP_MACRO -fopenmp -o test_gnu_omp saxpy_omp45_f.F90

gnu:
	gfortran -o test_gnu saxpy_omp45_f.F90

clean:
	rm -f *.o
	rm -f *.mod
	rm -f test_*
