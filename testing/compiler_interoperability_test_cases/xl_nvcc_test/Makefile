xlff90only:
	xlcuf -c saxpy_cuda_kernels.f90
	xlcuf -c test_saxpy_cuda_kernels.f90 
	xlf_r -c fmain.f90
	xlcuf fmain.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o -o fmain

xlflink:
	xlcuf -c saxpy_cuda_kernels.f90
	xlcuf -c test_saxpy_cuda_kernels.f90
	xlf_r -c fsubroutine.f90
	xlC_r -c c++_main.cc
	xlf -qcuda -o mixed_main c++_main.o fsubroutine.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o -lstdc++

xlclink:
	xlcuf -c saxpy_cuda_kernels.f90
	xlcuf -c test_saxpy_cuda_kernels.f90
	xlf_r -c fsubroutine.f90
	xlC_r -c c++_main.cc
	nvcc --compiler-bindir=xlC_r -Xcompiler -qsmp=omp -Xcompiler -qoffload -o mixed_main c++_main.o fsubroutine.o saxpy_cuda_kernels.o test_saxpy_cuda_kernels.o  -lxlf90_r -lxlcuf  -lcudart

clean:
	rm -f *.o
	rm -f *.mod
	rm -f fmain
