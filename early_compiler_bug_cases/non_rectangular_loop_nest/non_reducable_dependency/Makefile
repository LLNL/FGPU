xlf:
	xlf_r -g -qcheck -qsmp=omp -qoffload -qsigtrap test.F03
xlc:
	xlC_r -g -qcheck -qsmp=omp -qoffload test.cpp
gnu:
	gfortran -fopenmp test.F03
