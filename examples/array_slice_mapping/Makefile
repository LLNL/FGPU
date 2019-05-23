all:
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_array_slices_v1.f90 -o test_daxpy_array_slices_v1
	xlc_r -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -qsmp=omp -qoffload -qcuda daxpy_array_slices_v1.c -o test_daxpy_array_slices_v1_c
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_array_slices_v2.f90 -o test_daxpy_array_slices_v2
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_array_slices_v3.f90 -o test_daxpy_array_slices_v3
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_type_ptrs.f90 -o test_daxpy_type_ptrs

clean:
	rm -f *.o
	rm -f test_*
