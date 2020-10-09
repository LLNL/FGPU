cmake_minimum_required(VERSION 3.8)
project( derived_types_test LANGUAGES CXX Fortran)

set(BLT_SOURCE_DIR "/home/users/cmakrides/lib_builds/blt")

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------

# _blt_tutorial_include_blt_start
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT git submodule is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Default to C++11 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# include the basic directory, i.e. BLT specific stuff 
# Should be after the SetupBLT.cmake inclusion
if(NOT DEFINED FGPU_ROOT_DIR)
  set(FGPU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endIF()

if (EXISTS ${FGPU_ROOT_DIR}/commonFiles/BLT_Basics.cmake)
  include(${FGPU_ROOT_DIR}/commonFiles/BLT_Basics.cmake)
endif()

#All
set(SOURCE_FILES 
    main.f
    setup.f
    operations.f
    base_types.f)

foreach (_file ${SOURCE_FILES})
  set_property(SOURCE ${_file} APPEND_STRING PROPERTY COMPILE_FLAGS "${FORTRAN_USE_C_PREPROCESSOR} ${FORTRAN_FREE_FORMAT}")
endforeach()

set(programName runDerivedTypesTest)
blt_add_executable( NAME ${programName}
                    SOURCES ${SOURCE_FILES}
                    DEPENDS_ON openmp)


blt_add_target_link_flags(TO ${programName}
                          FLAGS -homp)

install(TARGETS ${programName}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
