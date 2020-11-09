set(target "omp_map_nested_data_structures")

add_executable(${target}
               nested_types.f90)

add_test(test_${target} ${target})
