all:
	xlf_r -qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace -o test openmp_tools.F90 nested_data.f03

clean:
	rm -f *.o *.mod test
