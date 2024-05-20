#CXX=xlc++_r
#CXXFLAGS=-qsmp=omp -qoffload
#CXX=craycxx
#CXXFLAGS=-fopenmp -O0 -g
CXX=icpx
CXXFLAGS=-fopenmp -O0 -g
#CXX=g++
#CXXFLAGS=-fopenmp -O0 -g
TARGET=saxpy
SRC=saxpy.cpp
OBJ=$(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean

