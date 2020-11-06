opt:
	xlf_r -DDISABLE_FORTRAN_MAIN -O3 -g1 -qsmp=omp -qoffload -c invalid_map_example.F90 -o invalid_map_example.o
	xlC_r -O3 -g1 -qsmp=omp -c main.cc -o main.o
	xlC_r -O3 -g1 -qsmp=omp -qoffload -o test_c_fortran main.o invalid_map_example.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath -lxlcufdevice

clang:
	xlf_r -DDISABLE_FORTRAN_MAIN -O3 -g1 -qsmp=omp -qoffload -c invalid_map_example.F90 -o invalid_map_example.o
	clang++ -O3 -g1 -fopenmp -c main.cc -o main.o
	xlC_r -O3 -g1 -qsmp=omp -qoffload -o test_c_fortran main.o invalid_map_example.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath -lxlcufdevice

clean:
	rm -f *.o
	rm -f test_*
