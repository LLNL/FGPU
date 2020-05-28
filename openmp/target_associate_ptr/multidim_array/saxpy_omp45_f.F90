subroutine testsaxpy_omp45_f

  use openmp_tools
  use iso_c_binding
  use omp_lib

  implicit none
  integer, parameter :: M = 2
  integer, parameter :: N = 2
  integer, parameter :: O = 2
  integer :: i, j, k, err
  logical(C_BOOL) :: use_external_allocator
 
  real(C_DOUBLE), pointer :: x(:,:,:), y(:,:,:)
  real(C_DOUBLE) :: a

  use_external_allocator = .TRUE.

  allocate( x(M,N,O), y(M,N,O) )
  x = 1.0
  y = 2.0
  a = 2.0

  call map_alloc(x, use_external_allocator)
  call map_alloc(y, use_external_allocator)

  !$omp target data map(to:N,a)
  !$omp target update to(x,y)

  ! Clear y on host to make sure GPU is really using the mapped y and not
  ! implicitly mapping it again.
  y=0.0

  ! Check that array shape information was copied to device by the
  ! target_associate_ptr call.
  !$omp target
  write(*,*) "--- check array contents on device, before kernel -----"
  write(*,*) "x(1,1,1) =", x(1,1,1)
  write(*,*) "x(2,2,2) =", x(2,2,2)
  write(*,*) "y(1,1,1) =", y(1,1,1)
  write(*,*) "y(2,2,2) =", y(2,2,2)
  write(*,*) "-------------------------------------------------------"
  !$omp end target

  !$omp target teams distribute parallel do private(i,j) shared(y,a,x) default(none) collapse(3)
  do i=1,M
    do j=1,N
      do k=1,N
        y(i,j,k) = a*x(i,j,k) + y(i,j,k)
      end do
    end do
  end do
  !$omp end target teams distribute parallel do
  !$omp end target data

  !$omp target
  write(*,*) "--- check array contents on device, after kernel -----"
  write(*,*) "x(1,1,1) =", x(1,1,1)
  write(*,*) "x(2,2,2) =", x(2,2,2)
  write(*,*) "y(1,1,1) =", y(1,1,1)
  write(*,*) "y(2,2,2) =", y(2,2,2)
  write(*,*) "------------------------------------------------------"
  !$omp end target

  !$omp target update from(y)
   
  write(*,*) "--- check array contents on host, after update from -----"
  write(*,*) "y(1,1,1) =", y(1,1,1)
  write(*,*) "y(2,2,2) =", y(2,2,2)
  write(*,*) "------------------------------------------------------"

  call map_delete(x, use_external_allocator)
  call map_delete(y, use_external_allocator)

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end subroutine testsaxpy_omp45_f
