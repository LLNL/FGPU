objects = setup.o derived_type.o
mpi = mpixlf
flags = -qoffload -qcuda -qsmp=omp -qfree=f90

gpudirect: main.o $(objects)
	$(mpi) -o $@ $(flags) $< $(objects)

main.o: main.f $(objects)
	$(mpi) -c $< $(flags)

derived_type.o: derived_type.f
	$(mpi) -c $< $(flags)

setup.o: setup.f derived_type.o
	$(mpi) -c $< $(flags)

clean:
	rm -f *.o *.mod gpudirect
