debug:
	xlf90_r -qcheck -qsigtrap -qsmp=noopt -g -qoffload -qcuda -o test_debug target_map_scalar.f90
opt:
	xlf90_r -qsmp=omp -O3 -qoffload -qcuda -o test_opt target_map_scalar.f90

clean:
	rm -f *.o test_*
