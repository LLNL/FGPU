# My best guess
all:
	xlcuf -qsmp=omp -qcuda -o test test.F90

clean:
	rm -f *.o mixed_lang_xlcuf mixed_lang_nvcc
	rm -f *.mod
	rm -f test_kernels
