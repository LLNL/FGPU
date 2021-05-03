pEXE = test_map_array_slices.exe

.PHONY: run
run: $(addprefix run_, $(basename $(pEXE)))

.PHONY: exe
exe: $(pEXE)

test_map_array_slices.exe: map_array_slices.f90
	-$(FC) $(FFLAGS)  map_array_slices.f90 -o test_map_array_slices.exe

run_%: %.exe
	-./$<

map:
	xlf_r -qsmp=omp -qoffload -qcuda map_array_slices.f90 -o test_map_array_slices
	xlf_r -qsmp=omp -qoffload -qcuda map_array_slices_in_type.f90 -o test_map_array_slices_in_type

update:
	xlf_r -qsmp=omp -qoffload -qcuda update_array_slices.f90 -o test_update_array_slices
	xlf_r -qsmp=omp -qoffload -qcuda update_array_slices_in_type.f90 -o test_update_array_slices_in_type
	xlf_r update_array_slices.f90 -o test_update_array_slices_cpu
	xlf_r update_array_slices_in_type.f90 -o test_update_array_slices_in_type_cpu

.PHONY: clean
clean:
	rm -rf $(pEXE) 
