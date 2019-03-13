# My best guess at reproducing the kind of build in MIRANDA and KULL.
all:
	xlf_r -g -qcuda -O0 -c saxpy.cuf -o saxpy_cuda_f.o
	xlf_r -g -qcuda -c fsubroutine.F90
	ar cr libfortranstuff.a fsubroutine.o saxpy_cuda_f.o
	xlC_r -g -qcuda -c main.cc
	xlC_r -g -qoffload -qcuda -o test_kernel main.o -L. -l fortranstuff -lxlf90_r -lxlcuf -lxlfmath

clean:
	rm -f *.o
	rm -f *.a
	rm -f *.mod
	rm -f test_kernel
