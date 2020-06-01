all:
	xlf_r -qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace -o test openmp_tools.f03 nested_data.f03

clean:
	rm -f *.o *.mod test
