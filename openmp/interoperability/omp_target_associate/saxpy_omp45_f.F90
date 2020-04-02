#define USE_CUSTOM_MAP

module openmp_device_memory_routines

   use iso_c_binding

   implicit none
   private

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr
! omp_target_memcpy

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

subroutine testsaxpy_omp45_f

  use openmp_device_memory_routines
  use iso_c_binding
  use omp_lib

  implicit none
!  integer, parameter :: N = ishft(1,21)
  integer, parameter :: N = 4
  integer :: i, err
 
  real, pointer :: x(:), y(:)
#if defined (USE_CUSTOM_MAP)
  real, pointer :: d_x(:), d_y(:)
  type(c_ptr) :: cptr, x_cptr, y_cptr
  integer(c_size_t) :: num_bytes, offset
#endif
  real :: a

  allocate( x(N), y(N) )
  x = 1.0
  y = 2.0
  a = 2.0

#if defined (USE_CUSTOM_MAP)
  num_bytes = sizeof(a)*N
  offset = 0

  x_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  err = omp_target_associate_ptr( C_LOC(x), x_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on x failed."
  endif

  y_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  err = omp_target_associate_ptr( C_LOC(y), y_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on y failed."
  endif
#endif

  !$omp target update to(x,y)
  !$omp target data map(to:N,a)

  !$omp target teams distribute parallel do private(i) shared(y,a,x) default(none)
  do i=1,N
    y(i) = a*x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do
  !$omp end target data

  !$omp target update from(y)

#if defined(USE_CUSTOM_MAP)
  err = omp_target_disassociate_ptr( C_LOC(x), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on x failed."
  endif
!  cptr = C_LOC(d_x)
  call omp_target_free( x_cptr, omp_get_default_device() )

  err = omp_target_disassociate_ptr( C_LOC(y), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on y failed."
  endif
!  cptr = C_LOC(d_y)
  call omp_target_free( y_cptr, omp_get_default_device() )
#endif

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end subroutine testsaxpy_omp45_f
