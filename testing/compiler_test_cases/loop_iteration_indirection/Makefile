xl:
	xlf_r -g -qcheck -qsmp=omp -qoffload -qsigtrap test.F03
gnu:
	gfortran -fopenmp test.F03 -ffree-form
cray:
	ftn -fopenmp test.F03
