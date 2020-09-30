all: opt debug debug2

opt:
	xlf2008_r -O3 -g -qsmp=omp -o test_opt test.f08
debug:
	xlf2008_r -O0 -g -qsmp=noopt -qcclines -o test_debug test.f08
debug2:
	xlf2008_r -O0 -g -qsmp=omp -qsmp=noopt -o test_debug2 test.f08
clean:
	rm test_*
