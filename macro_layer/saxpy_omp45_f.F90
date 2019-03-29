#include "macros.h"
program testsaxpy_omp45_f

  implicit none
  integer, parameter :: N = ishft(1,21)
  integer :: i
 
  real, allocatable :: x(:), y(:)
  real :: a

  allocate( x(N), y(N) )
  x = 1.0
  y = 2.0
  a = 2.0

OMP(target data map(to:N,a,x) map(tofrom: y))

OMP(target teams distribute parallel do private(i) shared(y,a,x) default(none))
  do i=1,N
    y(i) = a*x(i) + y(i)
  end do
OMP(end target teams distribute parallel do)

OMP(end target data)

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end program testsaxpy_omp45_f
