module openmp_device_memory_routines

   use iso_c_binding

   implicit none
   private

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr

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

      integer (c_int) function omp_target_associate_ptr( h_ptr, d_ptr, num_bytes, offset, device_num) bind ( c, name= 'omp_target_associate_ptr' )
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr, d_ptr
        integer(c_size_t), value :: num_bytes, offset
        integer(c_int), value :: device_num
      end function omp_target_associate_ptr

      integer (c_int) function omp_target_disassociate_ptr( h_ptr, device_num) bind ( c, name= 'omp_target_disassociate_ptr' )
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr
        integer(c_int), value :: device_num
      end function omp_target_disassociate_ptr

   end interface
end module openmp_device_memory_routines


subroutine testsaxpy_omp45_f

  use iso_c_binding
  use omp_lib
  use openmp_device_memory_routines
  implicit none

  integer :: N = 100
  integer :: i, j, k, err
  real(c_double) :: a = 2.0
 
  real(c_double), pointer :: x(:,:,:), y(:,:,:)
  type(c_ptr) :: x_cptr, y_cptr
  integer(c_size_t) :: num_bytes, offset

  !$omp target enter data map(to:N,a)

  allocate( x(N,N,N) )
  allocate( y(N,N,N) )

  x = 1.0
  y = 2.0

  call omp_set_default_device(0)

  num_bytes = sizeof(a)*N*N*N
  offset = 0

  x_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  y_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )

  err = omp_target_associate_ptr( C_LOC(x), x_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on x failed."
  endif

  err = omp_target_associate_ptr( C_LOC(y), y_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on y failed."
  endif

  !$omp target update to(x,y)
 
  ! Clear y on host
  y = 0.0 

  !$omp target teams distribute parallel do private(i,j,k) shared(y,a,x,N) default(none) collapse(3)
  do k=1,N
    do j=1,N
      do i=1,N
        y(i,j,k) = a*x(i,j,k) + y(i,j,k)
      end do
     end do
  end do
  !$omp end target teams distribute parallel do

  !$omp target update from(y)

  err = omp_target_disassociate_ptr( C_LOC(x), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on x failed."
  endif

  err = omp_target_disassociate_ptr( C_LOC(y), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on y failed."
  endif

  call omp_target_free( x_cptr, omp_get_default_device() )
  call omp_target_free( y_cptr, omp_get_default_device() )

  !$omp target exit data map(delete:N,a)

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))

end subroutine testsaxpy_omp45_f
