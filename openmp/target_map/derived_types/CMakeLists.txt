blt_add_executable( NAME test_derived_types
                    SOURCES main.f
                            setup.f
                            operations.f
                            base_types.f
                    DEPENDS_ON openmp)

install(TARGETS test_derived_types
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
