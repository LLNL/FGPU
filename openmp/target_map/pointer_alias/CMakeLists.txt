set(target "omp_map_pointer_alias")

add_executable(${target}
               ptr_to_data.f90)

install(TARGETS ${target} DESTINATION bin)
add_test(test_${target} ${target})
