blt_add_executable( NAME ptr_to_data
                    SOURCES ptr_to_data.f90
                    DEPENDS_ON openmp)

install(TARGETS ptr_to_data
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
