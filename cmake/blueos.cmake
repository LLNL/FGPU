# Host configuration file for IBM Power platforms with XL compiler

#------------------------------------------------------------------------------
# Compiler Definition
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER xlc_r CACHE PATH "")
set(CMAKE_CXX_COMPILER xlc++_r CACHE PATH "")
set(CMAKE_Fortran_COMPILER xlf2008_r CACHE PATH "")

# Override several of the results from FindOpenMP.cmake, as it doesn't support
# the right XL flags for target offloading (-qoffload is missing).
set(OpenMP_C_FLAGS -qsmp=omp -qoffload CACHE INTERNAL "")
set(OpenMP_CXX_FLAGS -qsmp=omp -qoffload CACHE INTERNAL "")
set(OpenMP_Fortran_FLAGS -qsmp=omp -qoffload CACHE INTERNAL "")

# Set some flags used in the build
set(Fortran_USE_C_PREPROCESSOR_FLAG -qpreprocess CACHE STRING "")

# This is a custom variables, not from FindOpenMP.cmake. FindOpenMP.cmake does
# not provide a OpenMP linker flag.  We need to inject some flags into the
# link line for XL.
set(OpenMP_Fortran_LINKER_FLAGS -qsmp=omp -qoffload CACHE INTERNAL "")

# Its not necessary to explicitly link in any omp libraries, the XL flags
# handle that automatically.
set(OpenMP_Fortran_LIBRARIES "" CACHE STRING "")
