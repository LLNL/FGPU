objects = operations.o setup.o base_types.o
xlcomp = xlf2003_r-gpu
flags = -qoffload

test: main.o $(objects)
	$(xlcomp) -o $@ $(flags) $< $(objects)

main.o: main.f $(objects)
	$(xlcomp) -c $< $(flags)

setup.o: setup.f base_types.o operations.o
	$(xlcomp) -c setup.f $(flags)

operations.o: operations.f base_types.o
	$(xlcomp) -c operations.f $(flags)
	
base_types.o: base_types.f
	$(xlcomp) -c base_types.f $(flags)
	
clean:
	rm -f *.o *.mod test
