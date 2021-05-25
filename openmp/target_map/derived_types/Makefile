objects = operations.o setup.o base_types.o

test: main.o $(objects)
	$(FC) -o $@ $(FFLAGS) $< $(objects)

main.o: main.f90 $(objects)
	$(FC) -c $< $(FFLAGS)

setup.o: setup.f90 base_types.o operations.o
	$(FC) -c setup.f90 $(FFLAGS)

operations.o: operations.f90 base_types.o
	$(FC) -c operations.f90 $(FFLAGS)

base_types.o: base_types.f90
	$(FC) -c base_types.f90 $(FFLAGS)

clean:
	rm -f *.o *.mod *.modmic test
