all:
	xlf_r -qsmp=omp -qoffload -qinfo=omperrtrace -g -O3 -o test example_types.F08 openmp_tools.F08 nested_data.F08

debug:
	xlf_r -qsmp=noopt -qoffload -qlomp_debug -qinfo=omperrtrace -g -O0 -o test example_types.F08 openmp_tools.F08 nested_data.F08

clean:
	rm -f *.o *.mod test
