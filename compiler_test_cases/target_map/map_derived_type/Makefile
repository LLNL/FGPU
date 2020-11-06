all:
	xlcuf -qsmp=omp -qoffload -c derived_type_mod.F90
	xlcuf -qsmp=omp -qoffload -c main.F90
	xlcuf -qsmp=omp -qoffload -o test main.o derived_type_mod.o

debug:
	xlcuf -g -O0 -qcheck -qsigtrap -qsmp=noopt -qoffload -c derived_type_mod.F90
	xlcuf -g -O0 -qcheck -qsigtrap -qsmp=noopt -qoffload -c main.F90
	xlcuf -g  -O0 -qcheck -qsigtrap  -qsmp=noopt -qoffload -o test main.o derived_type_mod.o

clean:
	rm -f *.o *.mod test
