program testsaxpy_omp45_f
  implicit none
  integer, parameter :: n = ishft(1,21)
  integer :: i
 
  real,allocatable :: x(:), y(:)
  real :: a

  allocate( x(n), y(n) )
  x = 1.0
  y = 2.0
  a = 2.0

  !$omp target data map(to:x) map(tofrom:y) use_device_ptr(x,y)
    call testsaxpy_cudafortran(x,y,a,n)
  !$omp end target data

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))
end program testsaxpy_omp45_f
