cmake_minimum_required(VERSION 3.8)

project( fgpu LANGUAGES C CXX Fortran)
include(CTest)

#-----------------------------------------------------------------------------
# Setup compiler flags for C++ and Fortran
#-----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

message(STATUS "Found OpenMP version ${OpenMP_VERSION}")
message(STATUS "Has OMP header? ${OpenMP_Fortran_HAVE_OMPLIB_HEADER}")
message(STATUS "Has OMPLIB module? ${OpenMP_Fortran_HAVE_OMPLIB_MODULE}")
message(STATUS "OpenMP_Fortran flags: ${OpenMP_Fortran_FLAGS}")
message(STATUS "OpenMP_Fortran libraries: ${OpenMP_Fortran_LIBRARIES}")

set(CMAKE_Fortran_FORMAT FREE)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${Fortran_USE_C_PREPROCESSOR_FLAG}")
message(STATUS "CMake Fortran flags: ${CMAKE_Fortran_FLAGS}")

#add_subdirectory(cuda)
#add_subdirectory(mpi)
add_subdirectory(openmp)
add_subdirectory(openmp_runtime_error_handling)
#add_subdirectory(preprocessor)
#add_subdirectory(test_cases)
