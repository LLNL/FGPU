test.exe: prim.f90 objects.f90 ompdata.f90 testmap.F90
	$(FC) $(FFLAGS) $^ -o $@ 

run_test:
	./test.exe

run: run_test

clean:
	rm -f *.o *.md *.mod test_* *.modmic *.exe
