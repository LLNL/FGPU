#FC=xlf2003_r
#FFLAGS=-qsmp=omp -qoffload

#FC=crayftn
#FFLAGS=-fopenmp -target-accel=amd_gfx942

FC=ifx
FFLAGS=-fopenmp
TARGET=saxpy
SRC=saxpy.f90
OBJ=$(SRC:.f90=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(FC) $(FFLAGS) -o $@ $^

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean

