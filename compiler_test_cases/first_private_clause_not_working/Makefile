debug:
	xlcuf -qcheck -qsigtrap -qsmp=noopt -g -qoffload -o test mc.f90
opt:
	xlcuf -qcheck -qsigtrap -qsmp=omp -O3 -qoffload -o test mc.f90

clean:
	rm -f *.o *.mod test
