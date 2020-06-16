all: opt debug

opt:
	xlf_r -qsmp=omp -qoffload -o test_opt nested_types.f03

debug:
	xlf_r -qsmp=noopt -O -g -qoffload -qlomp_debug -qinfo=omperrtrace -o test_debug nested_types.f03

clean:
	rm -f *.o *.mod test
