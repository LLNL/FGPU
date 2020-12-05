set(target "omp_map_nested_data_structures")

add_executable(${target}
               nested_types.f90)

install(TARGETS ${target} DESTINATION bin)
add_test(test_${target} ${target})
