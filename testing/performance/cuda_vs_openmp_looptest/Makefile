all:
	xlC_r -qsmp=omp -qoffload -c main.cc
	xlcuf -c looptest_cuda_mod.cuf
	xlcuf -qsmp=omp -qoffload -c looptest_omp45_mod.F90
	xlf -qsmp=omp -c fsubroutine.F90
	xlC_r -qoffload -qsmp=omp -qcuda -o test main.o fsubroutine.o looptest_omp45_mod.o looptest_cuda_mod.o -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib -lxlf90_r -lxlfmath -lxlcuf

clean:
	rm -f *.o *.mod test
