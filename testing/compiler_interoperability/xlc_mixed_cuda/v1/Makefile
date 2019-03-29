fortranonly:
	mpixlcuf -c saxpy_cuda_kernels.f90
	mpixlcuf -c test_saxpy_cuda_kernels.f90 
	mpixlf -c fmain.f90
	mpixlcuf fmain.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o -o fmain
	mpirun -n 2 ./fmain

xlflink:
	mpixlcuf -c saxpy_cuda_kernels.f90
	mpixlcuf -c test_saxpy_cuda_kernels.f90
	mpixlf -c fsubroutine.f90
	mpixlC -c c++_main.cc
	mpixlcuf -o mixed_main c++_main.o fsubroutine.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o -lstdc++
	mpirun -n 2 ./mixed_main

xlclink:
	mpixlcuf -c saxpy_cuda_kernels.f90
	mpixlcuf -c test_saxpy_cuda_kernels.f90
	mpixlf -c fsubroutine.f90
	mpixlC -c c++_main.cc
	nvcc --compiler-bindir=mpixlC -Xcompiler -qsmp=omp -Xcompiler -qoffload -o mixed_main c++_main.o fsubroutine.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o -lmpi_mpifh -lxlf90_r -lxlcuf -lcudart

clean:
	rm -f *.o
	rm -f *.mod
	rm -f fmain
