all:
	xlC_r -qsmp=omp -qoffload -c main.cc
	xlcuf -c looptest_cuda_mod.cuf
	xlcuf -qsmp=omp -qoffload -c looptest_omp45_mod.F90
	xlf -qsmp=omp -c fsubroutine.F90
	xlC_r -qoffload -qsmp=omp -qcuda -o test main.o fsubroutine.o looptest_omp45_mod.o looptest_cuda_mod.o -lxlf90_r -lxlfmath -lxlcuf

clean:
	rm -f *.o *.mod test
