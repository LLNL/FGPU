TARGET= Test

ALL: ${TARGET}

FLINKER = xlf2003_r	
#FFLAGS  = -qoffload -qsmp=noopt -qcuda -g -O0 -qcheck -qsigtrap -qstackprotect=all
FFLAGS  = -qoffload -qsmp=omp -g1 -O3 -qcuda

SOURCEF    = Test.f90 

OBJ = ${SOURCEF:.F90=.o}	

${TARGET}: ${OBJ}
	-${FLINKER} ${FFLAGS} -o ${TARGET} ${OBJ} 
clean:
	rm -rf ${TARGET}
	rm -rf *.mod
	rm -rf *.o
	rm -rf *~
