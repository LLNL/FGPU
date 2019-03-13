FULLDEBUG=-g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0

nvcc:
	nvcc -O3 -lineinfo -o saxpy_nvcc_opt saxpy_standalone.cu
	nvcc -O0 -G -o saxpy_nvcc_dbg saxpy_standalone.cu

# None of these kernels work in NVVP with source lines, even the NVCC compiled one...
xl:
#	nvcc -O3 -lineinfo -rdc=true -DNVCC -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c_opt.o
	nvcc -O3 -lineinfo -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c_opt.o
	xlC_r -g1 -O3 -c main.cc -o main_opt.o
	xlC_r -g1 -O3 -qoffload -qsmp=omp -qcuda -o saxpy_xl_opt main_opt.o saxpy_cuda_c_opt.o

#	nvcc -O0 -G -rdc=true -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c_dbg.o
	nvcc -O0 -G -arch sm_60 --compiler-bindir xlC_r -c saxpy_cuda_c.cu -o saxpy_cuda_c_dbg.o
	xlC_r -DCUDAC -DOMP45C -O0 -g $(FULLDEBUG)  -c main.cc -o main_dbg.o
	xlC_r $(FULLDEBUG) -o saxpy_xl_dbg main_dbg.o saxpy_cuda_c_dbg.o

clean:
	rm -f *.o 
	rm -f saxpy_nvcc_* saxpy_xl_*
