subroutine testsaxpy_omp45_f

  implicit none
  integer, parameter :: N = ishft(1,21)
  integer :: i
  real :: x(N), y(N), a
  
  x = 1.0
  y = 2.0
  a = 2.0

  !$omp target data map(to:N,a,x) map(from:y)

  !$omp target teams distribute parallel do private(i) shared(y,a,x) default(none)
  do i=1,N
    y(i) = a*x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do

  !$omp end target data

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end subroutine testsaxpy_omp45_f
