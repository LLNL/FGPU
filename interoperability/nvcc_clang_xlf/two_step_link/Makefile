SHELL = /bin/bash

xlroot = /usr/tce/packages/xl/xl-2020.03.18
cudaroot = /usr/tce/packages/cuda/cuda-10.1.243

cxx  = $(xlroot)/bin/xlc++
fc   = $(xlroot)/bin/xlf
nvcc = $(cudaroot)/bin/nvcc

openmp = -qsmp=omp
openmp_offload = -qoffload
fflg = -qsuppress=cmpmsg -qsuppress=1500-036 -qmaxmem=-1 -qtgtarch=sm_70 \
  -qxlf2003=polymorphic -qnosave -qfree=f90 -qsuffix=cpp=f:f=f

xl_libs = -L$(xlroot)/alllibs \
  -lomptarget-nvptx \
  -lxlsmp \
  -lxldevice \
  -lxlf90_r \
  -lxlfmath \
  -libmc++ \
  -lstdc++

COMPILE.f = $(fc) $(fflg) $(openmp) $(openmp_offload) 

COMPILE.cxx = $(nvcc) -std=c++11 -x cu -arch=sm_70 -I. -ccbin $(cxx)

objects = main.o particle.o v3.o quad.o problem.o

#v = -v

# Pattern rules for compiling
%.o %.mod : %.f
	$(COMPILE.f) $v -c $<

%.o: %.cpp
	$(COMPILE.cxx) $v -dc $< -o $@

# Link an executable.
app: $(objects)
	$(nvcc) $v -std=c++11 -arch=sm_70 -ccbin $(cxx) -dlink $(objects) $(xl_libs) -o foobar.o
	$(fc) $v $(openmp) $(openmp_offload) $(objects) foobar.o $(xl_libs) -o $@

.PHONY: clean
clean:
	rm -f *.o app *.mod
