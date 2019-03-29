default:
	xlc++_r -I$(CUDA_HOME)/include -std=c++11 -qsmp=omp -qoffload -qcuda main.cxx

clean:
	rm -f *.out *.o
