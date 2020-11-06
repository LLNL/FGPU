all:
	xlf_r -d -qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace -o test example_types.f08 openmp_tools.F08 nested_data.f08

debug:
	xlf_r -d -qsmp=noopt -qoffload -qlomp_debug -qinfo=omperrtrace -g -O0 -o test example_types.f08 openmp_tools.F08 nested_data.f08

clean:
	rm -f *.o *.mod test
