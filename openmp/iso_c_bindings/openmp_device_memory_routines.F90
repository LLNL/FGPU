module openmp_device_memory_routines

   use iso_c_binding

   implicit none
   private

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr
! omp_target_memcpy - TODO

   interface

      type(c_ptr) function omp_target_alloc( num_bytes, device_num ) bind ( c, name = 'omp_target_alloc' )
        use iso_c_binding
        implicit none

        integer(c_size_t), value :: num_bytes
        integer(c_int), value :: device_num
      end function omp_target_alloc

      subroutine omp_target_free( h_ptr, device_num ) bind ( c, name = 'omp_target_free' )
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr
        integer(c_int), value :: device_num
      end subroutine omp_target_free

      integer (c_int) function omp_target_associate_ptr( h_ptr, d_ptr, num_bytes, offset, device_num)
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr, d_ptr
        integer(c_size_t), value :: num_bytes, offset
        integer(c_int), value :: device_num
      end function omp_target_associate_ptr

      integer (c_int) function omp_target_disassociate_ptr( h_ptr, device_num)
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr
        integer(c_int), value :: device_num
      end function omp_target_disassociate_ptr

   end interface
end module openmp_device_memory_routines
