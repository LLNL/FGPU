
all:
	mpixlf -c fortran_hello.F
	mpixlC -c c++_main.cc
	mpixlC -o mixed_hello c++_main.o fortran_hello.o -L/opt/ibm/xlf/15.1.5/lib -lxlf90_r -lxlfmath -Wl,-rpath,/opt/ibm/xlf/15.1.5/lib -lmpi_mpifh
	mpirun -n 2 ./mixed_hello
