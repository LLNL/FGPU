# Host configuration file for IBM Power platforms with XL compiler

#------------------------------------------------------------------------------
# Setup some options
#------------------------------------------------------------------------------
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(ENABLE_OPENMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Compiler Definition
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER xlc_r CACHE PATH "")
set(CMAKE_CXX_COMPILER xlc++_r CACHE PATH "")
set(CMAKE_Fortran_COMPILER xlf2008_r CACHE PATH "")

#CMake will add -qsmp=omp, but doesn't know about -qoffload
set(OpenMP_C_FLAGS -qsmp=omp -qoffload CACHE PATH "")
set(OpenMP_CXX_FLAGS -qsmp=omp -qoffload CACHE PATH "")
set(OpenMP_Fortran_FLAGS -qsmp=omp -qoffload CACHE PATH "")

set(OpenMP_Fortran_LIB_NAMES "" CACHE PATH "")
set(OpenMP_Fortran_HAVE_OMPLIB_MODULE TRUE CACHE PATH "")
#set(BLT_OPENMP_LINK_FLAGS "-homp" CACHE STRING "") <- keep for cray example
#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")

#set(MPI_HOME             "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-6.0.0" CACHE PATH "")
set(MPI_C_COMPILER       "mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER     "mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER "mpixlf2008" CACHE PATH "")

set(MPIEXEC              "lrun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")
