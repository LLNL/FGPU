#include "macros.h"
subroutine testsaxpy_omp45_f

  use omp_lib
  use profilers_mod
  implicit none

  integer, parameter :: N = ishft(1,21)
  integer :: i
  
  real, pinned, allocatable :: x(:), y(:)
  real :: a

  allocate( x(N), y(N) )
  x = 1.0
  y = 2.0
  a = 2.0

START_RANGE("enter data map", 1)
  !$omp target enter data map(to: N, a, x, y)
END_RANGE()

START_RANGE("update to(a,x)", 2)
  !$omp target update to(a,x)
END_RANGE()

START_RANGE("kernel", 3)
  !$omp target teams distribute parallel do private(i) shared(y,a,x) default(none)
  do i=1,N
    y(i) = a*x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do
END_RANGE()

START_RANGE("update from", 4)
  !$omp target update from(y)
END_RANGE()

START_RANGE("exit data map", 5)
  !$omp target exit data map(delete: a, x, y)
END_RANGE()

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end subroutine testsaxpy_omp45_f
