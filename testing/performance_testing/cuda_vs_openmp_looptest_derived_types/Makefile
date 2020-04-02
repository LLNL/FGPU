all:
	xlcuf -c looptest_cuda_mod.cuf
	xlf -c derived_type_mod.F90
	xlcuf -qsmp=omp -qoffload -c looptest1_omp45_mod.F90
	xlcuf -qsmp=omp -qoffload -c looptest2_omp45_mod.F90
	xlcuf -qsmp=omp -c fsubroutine.F90
	xlC_r -qsmp=omp -qoffload -c main.cc
	xlC_r -qoffload -qsmp=omp -qcuda -o test main.o fsubroutine.o looptest1_omp45_mod.o looptest2_omp45_mod.o derived_type_mod.o looptest_cuda_mod.o -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib/ -lxlf90_r -lxlfmath -lxlcuf -lxlf90_r

clean:
	rm -f *.o *.mod test
