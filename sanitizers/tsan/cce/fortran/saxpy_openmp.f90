! This program performs a SAXPY operation using OpenMP
program saxpy_openmp
  use omp_lib
  implicit none

  integer, parameter :: N = 1000000
  real(kind=4) :: alpha = 2.0
  real(kind=4), dimension(N) :: X, Y
  integer :: i

  ! Initialize the arrays
  X = 1.0
  Y = 2.0

  ! Perform the SAXPY operation in parallel
  !$omp parallel do private(i) shared(X, Y, alpha)
  do i = 1, N
    Y(i) = alpha * X(i) + Y(i)
  end do
  !$omp end parallel do

  ! Print the result of the operation
  print *, 'The first element of Y after SAXPY is: ', Y(1)

end program saxpy_openmp
