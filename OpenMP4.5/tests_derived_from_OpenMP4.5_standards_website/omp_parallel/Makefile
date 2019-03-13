1a:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -o test omp_parallel.1a.f90
1acpu:
	xlcuf -qcheck -qsigtrap -qsmp=omp -o test omp_parallel.1a.f90

1b:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -o test omp_parallel.1b.f90
1bcpu:
	xlcuf -qcheck -qsigtrap -qsmp=omp -o test omp_parallel.1b.f90
clean:
	rm -f *.o *.mod test
