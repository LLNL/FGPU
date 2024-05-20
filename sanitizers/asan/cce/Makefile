# Define the compiler
FC = crayftn

# Define the target executable
TARGET = test_program

# Define objects
OBJECTS = math_operations.o test_program.o

# Define compile flags
FFLAGS = -O0 -g -fsanitize=address

# The first rule is the default rule
# Build the target executable
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $^

# Rule for building the module object
math_operations.o: math_operations.f90
	$(FC) $(FFLAGS) -c $<

# Rule for building the main program object
test_program.o: test_program.f90 math_operations.o
	$(FC) $(FFLAGS) -c $<

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)

# Declare phony targets
.PHONY: clean
