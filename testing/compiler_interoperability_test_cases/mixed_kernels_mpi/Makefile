# Both CUDA and OMP45 kernel, optimized
all:
	xlf_r -qcuda -O3 -c profilers_mod.F90 -o profilers_mod.o
	xlC_r -O3 -qcuda -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -qcuda -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlcuf -qextname=testsaxpy_cudafortran -O3 -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	mpicxx -DCUDAC -DCUDAF -DOMP45C -DOMP45F -O3 -qsmp=omp -qcuda -c main.cc
	mpicxx -O3 -qoffload -qsmp=omp -qcuda -o test_kernels profilers_mod.o main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lnvToolsExt -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_kernels
