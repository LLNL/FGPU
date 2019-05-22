all:
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_array_slices.f90 -o test_daxpy_array_slices

clean:
	rm -f *.o
	rm -f test_*
