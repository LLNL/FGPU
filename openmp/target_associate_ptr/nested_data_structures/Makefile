FC=ftn
CC=CC
LINK=ftn

OPTIONS_IBM=-qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace
FOPTIONS_CRAY=-homp -G0
#FOPTIONS_CRAY=-homp -hlist=a

FOPTIONS=$(FOPTIONS_CRAY)

OBJS=openmp_tools.o nested_data.o

#	xlf_r -qsmp=omp -qoffload -qlomp_debug -qinfo=omperrtrace -o test openmp_tools.f03 nested_data.f03

test: $(OBJS)
	$(FC) $(FOPTIONS) -o test $(OBJS)

nested_data.o nesred_data.mod: nested_data.f03 openmp_tools.mod
	$(FC) $(FOPTIONS) -c nested_data.f03

openmp_tools.o openmp_tools.mod: openmp_tools.f03
	$(FC) $(FOPTIONS) -c openmp_tools.f03

clean:
	rm -f *.o *.mod test *.cub *.lst
