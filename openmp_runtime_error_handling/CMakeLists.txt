message(STATUS "** OpenMP targets: applying openmp flags to all targets in /openmp_runtime_error_handling... **")
add_compile_options(${target} ${OpenMP_Fortran_FLAGS})
add_link_options(${target} ${OpenMP_Fortran_LIBRARIES} ${OpenMP_Fortran_LINKER_FLAGS})

add_subdirectory(overlapping_maps_error_case)
