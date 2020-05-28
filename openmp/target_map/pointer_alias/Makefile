default: serial threaded offload

serial:
	xlf_r -g ptr_to_data.f03 -o test_serial
threaded:
	xlf_r -g -qsmp=omp ptr_to_data.f03 -o test_threaded
offload:
	xlf_r -O3 -g -W@,-v -qlomp_debug -qinfo=omperrtrace -qsmp=omp -qoffload ptr_to_data.f03 -o test_offload

clean:
	rm -f *.mod
	rm -f test_*
