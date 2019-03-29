all:
	xlcuf -W@,-v -qsmp=omp -qoffload -o test workshare_example.f90
debug:
	xlcuf -W@,-v -g -O0 -qcheck -qsigtrap -qsmp=noopt -qoffload -o test workshare_example.f90
cpu:
	xlf_r -qsmp=omp -o test workshare_example.f90
cpu_debug:
	xlf_r -qsmp=omp -g -O0 -qcheck -qsigtrap -o cpu_test workshare_example.f90

clean:
	rm -f *.o *_test *.core
