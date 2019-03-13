pinned:
	mpixlf -qcheck -qsigtrap -qsmp=omp -qoffload -o test_pinned pinned_ptr.F90

pinned_dealloc:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qcuda -o test_pinned_dealloc pinned_dealloc.F90

pinnedreg:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qcuda -o test_pinned_reg pinned_register_ptr.F90

notpinned:
	xlcuf -qcheck -qsigtrap -qsmp=omp -qoffload -qcuda -o test_not_pinned not_pinned_ptr.F90

pinnedalloc:
	mpixlf -qcheck -qsigtrap -qsmp=omp -qoffload -o test_pinned_alloc pinned.F90
