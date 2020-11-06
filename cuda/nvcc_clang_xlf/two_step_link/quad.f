module quad
  use iso_c_binding 
  use omp_lib
  implicit none

contains

! Compute the definite integral of 50/(pi*(2500*x**2+1)) from 0 to 10.
! Exact answer is atan(500)/pi ~ 0.499363.
function f_quad() bind(c)
  use iso_c_binding 
  use omp_lib
  implicit none
  real(c_double) :: f_quad
  real(c_double) :: a=0.0, b=10.0, total=0.0, x
  real(c_double), parameter :: pi = 3.141592653589793D+00
  integer :: i
  integer,parameter :: n=10000000

!$omp target teams distribute parallel do reduction(+:total)
  do i = 1, n
    x = ( real(n-i,c_double)*a + real(i-1,c_double)*b ) / real(n-1,c_double)
    total = total + 50.0D+00 / (pi*(2500.0D+00*x*x+1.0D+00))
  end do
!$omp end target teams distribute parallel do     

  total = (b - a) * total/real(n,c_double)
  f_quad = total
end function

end module
