all: serial threaded offloaded

serial:
	mpixlf -g prim.f03 objects.f03 ompdata.f03 testmap.f03 -o test_cpu_serial

threaded:
	mpixlf -g -qsmp=omp -qinfo=omperrtrace prim.f03 objects.f03 ompdata.f03 testmap.f03 -o test_cpu_threaded

offloaded:
	mpixlf -g -qinfo=omperrtrace -qsmp=omp -qoffload prim.f03 objects.f03 ompdata.f03 testmap.f03 -o test_gpu

clean:
	rm -f *.o
	rm -f *.mod
	rm -f test_*
