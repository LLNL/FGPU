FULLDEBUG=-g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0

nvcc_opt:
	nvcc -O3 -o saxpy_nvcc saxpy_nvcc_only.cu

# Works fine in NVVP with source lines
nvcc_debug:
	nvcc -O3 -lineinfo -o saxpy_nvcc_debug saxpy_nvcc_only.cu

# Both CUDA and OMP45 kernel, optimized
opt:
	xlC_r -O3 -g1 -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -g1 -qextname=testsaxpy_omp45_f -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlf_r -d -g1 -qextname=testsaxpy_cudafortran -O3 -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	nvcc -O3 -lineinfo -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -g1 -DCUDAC -DCUDAF -DOMP45C -DOMP45F -O3 -qsmp=omp -c main.cc
	xlC_r -g1 -O3 -qoffload -qsmp=omp -qcuda -o test_kernels_opt main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

# Both CUDA and OMP45 kernels, debug
debug:
	xlC_r  $(FULLDEBUG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(FULLDEBUG) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlf_r -d -qextname=testsaxpy_cudafortran $(FULLDEBUG) -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	nvcc -O0 -lineinfo -G -rdc=true -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -DCUDAC -DCUDAF -DOMP45C -DOMP45F -O0 -g $(FULLDEBUG)  -c main.cc
	xlC_r $(FULLDEBUG) -o test_kernels_debug main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_kernels_*
