all:
	xlf_r -qsmp=omp -qoffload -qinfo=omperrtrace -o test example_types.f08 openmp_tools.f08 nested_data.f08

debug:
	xlf_r -qsmp=noopt -qoffload -qlomp_debug -qinfo=omperrtrace -g -O0 -o test example_types.f08 openmp_tools.f08 nested_data.f08

clean:
	rm -f *.o *.mod test
