# Makefile for SAXPY OpenMP Fortran Program

# Compiler
FC = crayftn
# Compiler flags
FCFLAGS = -fopenmp -fsanitize=thread -g

all: saxpy_openmp saxpy_openmp_race
# Build target
saxpy_openmp: saxpy_openmp.f90
	$(FC) $(FCFLAGS) -o $@ $^

saxpy_openmp_race: saxpy_openmp_race.f90
	$(FC) $(FCFLAGS) -o $@ $^
# Clean target
clean:
	rm -f saxpy_openmp saxpy_openmp_race *.mod

# Phony targets
.PHONY: clean
