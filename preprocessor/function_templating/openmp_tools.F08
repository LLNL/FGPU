module openmp_tools

   use iso_c_binding
   use omp_lib
   use example_types

   implicit none

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr

   interface

      type(C_PTR) function omp_target_alloc( num_bytes, device_num ) bind ( c, name = 'omp_target_alloc' )
        use iso_c_binding
        implicit none

        integer(C_SIZE_T), value :: num_bytes
        integer(C_INT), value :: device_num
      end function omp_target_alloc

      subroutine omp_target_free( h_ptr, device_num ) bind ( c, name = 'omp_target_free' )
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr
        integer(C_INT), value :: device_num
      end subroutine omp_target_free

      integer (C_INT) function omp_target_associate_ptr( h_ptr, d_ptr, num_bytes, offset, device_num)
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr, d_ptr
        integer(C_SIZE_T), value :: num_bytes, offset
        integer(C_INT), value :: device_num
      end function omp_target_associate_ptr

      integer (C_INT) function omp_target_disassociate_ptr( h_ptr, device_num)
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr
        integer(C_INT), value :: device_num
      end function omp_target_disassociate_ptr

   end interface

contains

! Instantiate functions for double arrays of 1,2,3 ranks
#define OMPFTL_TYPE1 real
#define OMPFTL_TYPE2 C_DOUBLE
#define OMPFTL_RANK 1
#include "map_functions.F90_template"

#define OMPFTL_TYPE1 real
#define OMPFTL_TYPE2 C_DOUBLE
#define OMPFTL_RANK 2
#include "map_functions.F90_template"

#define OMPFTL_TYPE1 real
#define OMPFTL_TYPE2 C_DOUBLE
#define OMPFTL_RANK 3
#include "map_functions.F90_template"

! Instantiate functions for derived types
#define OMPFTL_TYPE1 type
#define OMPFTL_TYPE2 typeS
#define OMPFTL_RANK 1
#include "map_functions.F90_template"

#define OMPFTL_TYPE1 type
#define OMPFTL_TYPE2 typeQ
#define OMPFTL_RANK 0
#include "map_functions.F90_template"

end module openmp_tools
