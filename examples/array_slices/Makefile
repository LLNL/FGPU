all: map update

map:
	xlf_r -qsmp=omp -qoffload -qcuda map_array_slices_upfront.f90 -o test_map_array_slices_upfront
	xlc_r -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -qsmp=omp -qoffload -qcuda map_array_slices_upfront.c -o test_map_array_slices_upfront_c
	xlf_r -qsmp=omp -qoffload -qcuda map_array_slices.f90 -o test_map_array_slices
	xlf_r -qsmp=omp -qoffload -qcuda map_array_slices_in_type.f90 -o test_map_array_slices_in_type

update:
	xlf_r -qsmp=omp -qoffload -qcuda update_array_slices.f90 -o test_update_array_slices
	xlf_r -qsmp=omp -qoffload -qcuda update_array_slices_in_type.f90 -o test_update_array_slices_in_type

clean:
	rm -f *.o
	rm -f test_*
