all: map update

map:
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_map_array_slices_upfront.f90 -o test_daxpy_map_array_slices_upfront
	xlc_r -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -qsmp=omp -qoffload -qcuda daxpy_map_array_slices_upfront.c -o test_daxpy_map_array_slices_upfront_c
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_map_array_slices_inline.f90 -o test_map_array_slices_inline
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_map_types.f90 -o test_daxpy_map_types
	
update:
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_update_array_slices.f90 -o test_daxpy_update_array_slices
	xlf_r -qsmp=omp -qoffload -qcuda daxpy_update_array_slices_in_type.f90 -o test_daxpy_update_array_slices_in_type

clean:
	rm -f *.o
	rm -f test_*
