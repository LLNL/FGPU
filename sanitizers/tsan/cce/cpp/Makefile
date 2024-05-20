# Makefile for SAXPY OpenMP C++ Program

# Compiler
CXX = craycxx
# Compiler flags
CXXFLAGS = -fopenmp -fsanitize=thread
# Target
TARGET = saxpy_openmp
# Source files
SRC = saxpy_openmp.cpp

# Build target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean target
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: clean
