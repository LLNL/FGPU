all:
	xlcuf -c -qsmp=omp -qoffload derived_type_mod.cuf
	xlcuf -c -qsmp=omp -qoffload cuda_test_mod.cuf
	xlcuf -qsmp=omp -qoffload -c main.F90
	xlcuf -qcuda -qoffload -qsmp=omp -o test derived_type_mod.o cuda_test_mod.o main.o

clean:
	rm -f *.o *.mod test
