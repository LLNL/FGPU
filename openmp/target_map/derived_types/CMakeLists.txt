set(target "omp_map_derived_types")

add_executable(${target}
               main.f
               setup.f
               operations.f
               base_types.f)

install(TARGETS ${target} DESTINATION bin)
add_test(test_${target} ${target})
