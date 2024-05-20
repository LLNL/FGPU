# Makefile for SAXPY OpenMP C++ Program

# Compiler
CXX = craycxx
# Compiler flags
CXXFLAGS = -fopenmp -fsanitize=thread -g
# Source files
SAXPY_OPENMP_SRC = saxpy_openmp.cpp
SAXPY_OPENMP_RACE_SRC = saxpy_openmp_race.cpp

ALL: saxpy_openmp saxpy_openmp_race

# Build target
$(SAXPY_OPENMP): saxpy_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

$(SAXPY_OPENMP_RACE): saxpy_openmp_race.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean target
clean:
	rm -f saxpy_openmp saxpy_openmp_race *.o

# Phony targets
.PHONY: clean
