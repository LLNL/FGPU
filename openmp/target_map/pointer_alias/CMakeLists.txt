set(target "omp_map_pointer_alias")

add_executable(${target}
               ptr_to_data.f90)

add_test(test_${target} ${target})
