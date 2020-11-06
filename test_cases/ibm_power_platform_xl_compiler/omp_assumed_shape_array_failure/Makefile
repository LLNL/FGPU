# Only CUDA kernels
cuda:
	xlcuf -O3 -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -DCUDAC -DCUDAF -O3 -qsmp=omp -c main.cc
	xlC_r -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_cuda_c.o -lxlf90_r -lxlcuf -lxlfmath

# Both CUDA and OMP45 kernel, optimized
allopt:
	xlC_r -O3 -qsmp=omp -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -O3 -qsmp=omp -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlcuf -O3 -c saxpy_cuda_f.cuf -o saxpy_cuda_f.o
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -DCUDAC -DCUDAF -DOMP45C -DOMP45F -O3 -qsmp=omp -c main.cc
	xlC_r -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlcuf -lxlfmath

# Both CUDA and OMP45 kernels, debug
alldebug:
	xlC_r -O0 -g -qsmp=noopt -qoffload -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -O0 -g -qsmp=noopt -qoffload -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlcuf -O0 -g -c saxpy_omp45_f.cuf -o saxpy_cuda_f.o
	nvcc -O0 -g -rdc=true -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c.o
	xlC_r -DCUDAC -DCUDAF -DOMP45C -DOMP45F -O0 -g -qsmp=noopt -c main.cc
	xlC_r -O0 -g -qoffload -qsmp=noopt -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_omp45_f.o saxpy_cuda_c.o -lxlf90_r -lxlcuf -lxlfmath


# Only cuda fortran kernel.
pgi:
	pgf90 -O3 -Mcuda -c saxpy.cuf -o saxpy_cuda_f.o
	pgc++ -DCUDAF -O3 -mp -c main.cc
	pgc++ -O3 -mp -Mcuda -o test_kernels main.o saxpy_cuda_f.o -lpgf90rtl -lpgf90 -lpgf902 -lcudafor -lcudafor2 -lpgf90_rpm1 -lcudaforblas

clean:
	rm -f *.o 
	rm -f *.mod
	rm -f test_kernels
