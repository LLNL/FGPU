xlf:
	xlf_r -O3 -qxlf90=signedzero,autodealloc,nooldpad -qxlf2003=autorealloc,bozlitargs,nodynamicacval,nooldnaninf,polymorphic,signdzerointr,stopexcept,volatile,polymorphic -c memory_allocators_mod.F90 -o memory_allocators_mod.o
	xlf_r -O3 -qxlf90=signedzero,autodealloc,nooldpad -qxlf2003=autorealloc,bozlitargs,nodynamicacval,nooldnaninf,polymorphic,signdzerointr,stopexcept,volatile,polymorphic -c driver.F90 -o driver.o
	xlf_r -O3 -qoffload -qcuda -o test driver.o memory_allocators_mod.o

xlfcuda:
	xlf_r -O3 -DCUDA  -qxlf90=signedzero,autodealloc,nooldpad -qxlf2003=autorealloc,bozlitargs,nodynamicacval,nooldnaninf,polymorphic,signdzerointr,stopexcept,volatile,polymorphic -c memory_allocators_mod.F90 -o memory_allocators_mod.o
	xlf_r -O3 -qxlf90=signedzero,autodealloc,nooldpad -qxlf2003=autorealloc,bozlitargs,nodynamicacval,nooldnaninf,polymorphic,signdzerointr,stopexcept,volatile,polymorphic -c driver.F90 -o driver.o
	xlf_r -O3 -qoffload -qcuda -o test driver.o memory_allocators_mod.o

pgi:
	pgfortran -O3 -c memory_allocators_mod.F90 -o memory_allocators_mod.o
	pgfortran -O3 -c driver.F90 -o driver.o
	pgfortran -O3 -o test driver.o memory_allocators_mod.o

pgicuda:
	pgfortran -O3 -c -DUSE_CUDA memory_allocators_mod.F90 -o memory_allocators_mod.o
	pgfortran -O3 -c driver.F90 -o driver.o
	pgfortran -O3 -Mcuda -o test driver.o memory_allocators_mod.o

gnu:
	gfortran -O3 -c memory_allocators_mod.F90 -o memory_allocators_mod.o
	gfortran -O3 -c driver.F90 -o driver.o
	gfortran -O3 -o test driver.o memory_allocators_mod.o

clean:
	rm -f *.o test
	rm -f *.mod
