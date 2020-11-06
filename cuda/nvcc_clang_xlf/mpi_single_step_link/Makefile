all: xlf_link xlc_link

xlf_link:
	mpiclang++ -O2 -g -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlf -g -O2 -qoffload -qsmp=omp -qcuda -o test_xlf_link main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlfmath

xlf_link_mpip:
	mpiclang++ -g -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlf -g -O2 -qoffload -qsmp=omp -qcuda -o test_xlf_link_mpip main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlfmath -lmpiP

xlf_link_ppt:
	mpiclang++ -g -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlf -g -O2 -qoffload -qsmp=omp -qcuda -o test_xlf_link_ppt main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlfmath -L${IHPCT_BASE}/lib64 -R${IHPCT_BASE}/lib64 -lmpitrace -lotf2trace

xlc_link:
	mpiclang++ -g -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlC -g -O2 -qoffload -qsmp=omp -o test_xlc_link main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlfmath

xlc_link_mpip:
	mpiclang++ -g -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlC -lmpip -g -O2 -qoffload -qsmp=omp -o test_xlc_link_mpip main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlfmath

xlc_link_ppt:
	mpiclang++ -g -O2 -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	mpixlf -g -qextname=testsaxpy_omp45_f -O2 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	nvcc -g --compiler-bindir clang++ --compiler-options -O2 --compiler-options -fopenmp -O2  -arch sm_70 -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpiclang++ -g -O2 -fopenmp -c main.cc
	mpixlC -g -O2 -qoffload -qsmp=omp -o test_xlc_link_ppt main.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -L/usr/tce/packages/xl/xl-2020.03.18/xlf/16.1.1/lib -lxlf90_r -lxlfmath -L${IHPCT_BASE}/lib64 -R${IHPCT_BASE}/lib64 -lmpitrace -lotf2trace

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_*
