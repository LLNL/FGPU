all: opt debug debug2

opt:
	xlf2008_r -O3 -g -qsmp=omp -o test_opt test.f08
# The -qicclines (enables OpenMP conditional source) appears to be disabled by the -qsmp=noopt, regardless
# of order of flags in compile line.
debug:
	xlf2008_r -O0 -g -qsmp=noopt -qicclines -o test_debug test.f08
debug2:
	xlf2008_r -O0 -g -qsmp=omp -qsmp=noopt -o test_debug2 test.f08
clean:
	rm test_*
