blt_add_executable( NAME map_overlap
                    SOURCES map_overlap.f90
                    DEPENDS_ON openmp)

get_filename_component(parentDir ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
get_filename_component(parentDir ${parentDir} NAME)

install(TARGETS map_overlap
        RUNTIME DESTINATION bin/${parentDir}
        LIBRARY DESTINATION lib)
