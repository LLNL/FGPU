SRC = $(wildcard *.f90)
EXE =  $(SRC:%.f90=test_%.exe)

.PHONY: run_all
run_all: $(addprefix run_, $(basename $(EXE)))

test_%.exe: %.f90
	-$(FC) $(FFLAGS) $< -o $@

run_%: %.exe
	-./$<

.PHONY: clean
clean:
	rm -rf $(EXE) 
