1:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1.f90

1b:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1b.f90

1c:
	xlcuf -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1c.f90

1d:
	xlcuf -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1d.f90

1e:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1e.f90

1f:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_target_unstructured_data.1f.f90

foo:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test foo.f90

clean:
	rm -f *.o *.mod test
