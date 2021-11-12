blt_add_executable( NAME nested_data_structures
                    SOURCES nested_types.f90
                    DEPENDS_ON openmp)

install(TARGETS nested_data_structures
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
