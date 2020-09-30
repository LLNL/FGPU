cpu:
	xlf2008_r -O3 -g -qsmp=omp -W@,-v -o max_cpu max.f08
gpu2:
	xlf2008_r -O2 -g -qsmp=omp -qoffload -W@,-v -o max_gpu2 max.f08
gpu3:
	xlf2008_r -O3 -g -qsmp=omp -qoffload -W@,-v -o max_gpu3 max.f08
clean:
	rm *.mod
	rm max_*
