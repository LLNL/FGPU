This is a collection of code examples focusing on porting FORTRAN codes to run on GPUs.  The purpose of these is to provide both learning aids for developers and a regression test for compilers supporting OpenMP4.5, OpenACC, CUDA FORTRAN, etc.

While these examples have a heavy FORTRAN emphasis, some examples also include C++ usage.

These include:
CUDA - CUDA examples
debug_symbols_issues - Examples of compiling device code with debug symbols to work with debuggers
kernel_scaling_tests - early performance tests
memory_testing - tests of pinned memory, unified memory, etc
mixed_language_trials - many codes are C++ and FORTRAN executables.  This contains examples of trials compiling mixed language examples and launching kernels from different languages/technologies in the same executable.
OpenMP4.5 - tests exercising OpenMP pragmas
performance_issues - early trials of performance testing with specific features

Sample Makefiles are included which support the LLNL Sierra platform using the IBM XL compiler.

As an initial compiler regression test, it is strongly advised to first compile and test the OpenMP4.5 examples at:
https://github.com/OpenMP/Examples/tree/v4.5.0

RELEASE
LLNL-CODE-769479
