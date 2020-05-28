# My best guess

CC=CC
FC=ftn
CCg=nvcc
FCg=ftn

#OpenMP flags
# Cray: -homp / -fopenmp
# pgi: -mp 

nvidiaTarget=$(subst nvidia,,${CRAY_ACCEL_TARGET})

COPTIONS=-mp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_$(nvidiaTarget) -O3
FOPTIONS=-mp -O3
COPTIONS_GPU=-rdc=true -DNVCC -arch sm_$(nvidiaTarget) --compiler-bindir $(CC)

CINCLUDE=-I$(CUDA_HOME)/include
FINCLUDE=


all:
	$(CC)  $(COPTIONS) $(CINDLUDE) -c saxpy.cc -o saxpy_omp45_c.o
	$(FCg) $(FOPTIONS) -c saxpy.cuf -o saxpy_cuda_f.o
	$(CCg) -O3 $(COPTIONS_GPU) -c saxpy.cu -o saxpy_cuda_c.o
	$(CC) $(COPTIONS) $(CINCLUDE) -c main.cc
	$(CC) -O3 -qoffload -qsmp=omp -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib/ -lxlf90_r -lxlcuf -lxlfmath

debug:
	xlC_r -qcheck -O0 -g -qsmp=noopt -qoffload -c saxpy.cc -o saxpy_omp45_c.o
	xlcuf -qcheck -O0 -g -c saxpy.cuf -o saxpy_cuda_f.o
	nvcc -O0 -g -rdc=true -DNVCC -arch sm_70 --compiler-bindir xlC_r -c saxpy.cu -o saxpy_cuda_c.o
	xlC_r -qcheck -O0 -g -qsmp=noopt -c main.cc
	xlC_r -qcheck -O0 -g -qoffload -qsmp=noopt -qcuda -o test_kernels main.o saxpy_cuda_f.o saxpy_omp45_c.o saxpy_cuda_c.o  -L/usr/tce/packages/xl/xl-2019.12.23/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o mixed_lang_xlcuf mixed_lang_nvcc
	rm -f *.mod
	rm -f test_kernels
