all:
	xlf_r -qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace -o test example_types.f03 openmp_tools.f03 nested_data.f03

debug:
	xlf_r -qsmp=noopt -qoffload -qlomp_debug -qinfo=omperrtrace -g -O0 -o test example_types.f03 openmp_tools.f03 nested_data.f03

clean:
	rm -f *.o *.mod test
