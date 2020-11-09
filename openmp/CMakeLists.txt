message(STATUS "** OpenMP targets: applying openmp flags to all targets in /openmp... **")
add_compile_options(${target} ${OpenMP_Fortran_FLAGS})
add_link_options(${target} ${OpenMP_Fortran_LIBRARIES} ${OpenMP_Fortran_LINKER_FLAGS})

add_subdirectory(target_map)
