all:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -o test double_free.f90
debug:
	xlcuf -g -O0 -qcheck -qsigtrap -qsmp=noopt -qoffload -o test double_free.f90
cpu:
	xlcuf -g -O0 -qcheck -qsigtrap -o test double_free.f90
gfortran:
	gfortran -g -O0 -o test double_free.f90

clean:
	rm -f *.o *.mod test *.core
