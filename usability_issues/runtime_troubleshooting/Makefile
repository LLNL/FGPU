all:
	xlcuf -W@,-v -qsmp=omp -qoffload -o test_map_overlap map_overlap.f90
	xlcuf -W@,-v -qsmp=omp -qoffload -o test_out_of_bounds out_of_bounds.f90
debug:
	xlcuf -W@,-v -g -O0 -qlomp_debug -qinfo=omperrtrace -qcheck -qsigtrap -qsmp=noopt -qoffload -o test_map_overlap map_overlap.f90
	xlcuf -W@,-v -g -O0 -qlomp_debug -qinfo=omperrtrace -qcheck -qsigtrap -qsmp=noopt -qoffload -o test_out_of_bounds out_of_bounds.f90
cpu:
	xlf_r -qsmp=omp -o test_cpu_map_overlap map_overlap.f90
	xlf_r -qsmp=omp -o test_cpu_out_of_bounds out_of_bounds.f90
cpu_debug:
	xlf_r -qsmp=omp -qlomp_debug -qinfo=omperrtrace -g -O0 -qcheck -qsigtrap -o test_cpu_map_overlap map_overlap.f90
	xlf_r -qsmp=omp -qlomp_debug -qinfo=omperrtrace -g -O0 -qcheck -qsigtrap -o test_cpu_out_of_bounds out_of_bounds.f90

clean:
	rm -f *.o test_* *.core
