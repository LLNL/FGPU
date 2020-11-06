1a:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_array_issues.1a.f90
1b:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qxlf2003=nopolymorphic -o test Example_array_issues.1b.f90
1acpu:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qxlf2003=nopolymorphic -o test Example_array_issues.1a.f90

clean:
	rm -f *.o *.mod test
