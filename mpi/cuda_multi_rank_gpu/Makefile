all:
	nvcc -O3 -rdc=true -DNVCC -arch sm_60 --compiler-bindir mpicxx -c saxpy.cu -o saxpy_cuda_c.o
	mpicxx -O3 -I$(CUDA_HOME)/include -c main.cc
	mpicxx -O3 -qcuda -o test_kernels main.o saxpy_cuda_c.o -L /usr/tce/packages/cuda/cuda-10.1.243/lib64 -lcudart -lcudadevrt

clean:
	rm -f test_kernels *.o
