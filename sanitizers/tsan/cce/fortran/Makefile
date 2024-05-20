# Makefile for SAXPY OpenMP Fortran Program

# Compiler
FC = crayftn
# Compiler flags
FCFLAGS = -fopenmp -fsanitize=thread
# Target
TARGET = saxpy_openmp
# Source files
SRC = saxpy_openmp.f90

# Build target
$(TARGET): $(SRC)
	$(FC) $(FCFLAGS) -o $@ $^

# Clean target
clean:
	rm -f $(TARGET) *.mod

# Phony targets
.PHONY: clean
