all:
	mpicxx test.c -o test
	mpicxx -qsmp=omp -qoffload test_omp.c -o test_omp

clean:
	rm test test_omp
