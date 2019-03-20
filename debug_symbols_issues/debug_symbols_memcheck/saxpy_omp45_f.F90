subroutine testsaxpy_omp45_f

  implicit none
  integer, parameter :: N = ishft(1,21)
  integer :: i
! segfaults, out of memory stack issue?
!  real :: x(N), y(N), a
 
  real, allocatable :: x(:), y(:)
  real :: a

  allocate( x(N), y(N) )
  x = 1.0
  y = 2.0
  a = 2.0

  !$omp target data map(to:N,a,x) map(tofrom: y)

  !$omp target teams distribute parallel do private(i) shared(y,a,x) default(none)
  do i=1,N
    ! Add 1000 to create an out of bounds memory access to test cuda-memcheck
    y(i+1000) = a*x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do

  !$omp end target data

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end subroutine testsaxpy_omp45_f
