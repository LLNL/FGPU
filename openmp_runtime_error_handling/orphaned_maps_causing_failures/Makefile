default: serial threaded offload

serial:
	xlf_r -g orphaned_array_map.f03 -o test_arr_serial
	xlf_r -g orphaned_ptr_map.f03 -o test_ptr_serial
threaded:
	xlf_r -g -qsmp=omp orphaned_array_map.f03 -o test_arr_threaded
	xlf_r -g -qsmp=omp orphaned_ptr_map.f03 -o test_ptr_threaded
offload:
	xlf_r -O3 -g -W@,-v -qinfo=omperrtrace -qsmp=omp -qoffload orphaned_array_map.f03 -o test_arr_offload
	xlf_r -O3 -g -W@,-v -qinfo=omperrtrace -qsmp=omp -qoffload orphaned_ptr_map.f03 -o test_ptr_offload

clean:
	rm -f *.mod
	rm -f test_*
