# Host configuration file for redwood using cce 
# Assuming that the PrgEnv-cray module is loaded and an accelerator module is also loaded for openMP offload 

#------------------------------------------------------------------------------
# Compiler Definition
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "ftn" CACHE PATH "")
