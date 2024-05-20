CXX=xlc++_r
CXXFLAGS=-qsmp=noopt -qoffload -O0 -g -qcheck=all
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

