all:
	xlf_r -g -qsmp=omp -qoffload -c derived_type_mod.F90
	xlf_r -g -qsmp=omp -qoffload -c kernel_omp45_mod.F90
	xlf_r -g -qsmp=omp -qoffload -c main.F90
	xlf_r -g -qsmp=omp -qoffload -o test main.o kernel_omp45_mod.o derived_type_mod.o

serial:
	xlf -c derived_type_mod.F90
	xlf -c kernel_omp45_mod.F90
	xlf -c main.F90
	xlf -o test main.o kernel_omp45_mod.o derived_type_mod.o

clean:
	rm -f *.o *.mod test
