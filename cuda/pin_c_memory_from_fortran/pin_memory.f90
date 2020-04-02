! This example registers (pins) an existing array in FORTRAN.
program pin_array
   use cudafor
   use iso_c_binding
   implicit none

   integer j, istat
   real, pointer, contiguous, dimension(:) :: x
   real, allocatable, target, dimension(:) :: y
   real :: dummy

   allocate(x(1024))
   allocate(y(1024))

   x(:) = 1.0
   y(:) = 2.0

   istat = cudaHostRegister(x, 1024*SIZEOF(dummy), 0)
   istat = cudaHostRegister(y, 1024*SIZEOF(dummy), 0)

   print *, x(1:10)
   print *, y(1:10)
   print *, "---------------------"
   print *, "Multiple by 2 on GPU"
   print *, "---------------------"

   !$omp target map (tofrom:x, y)

   x(1:10) = 2.0
   y(1:10) = 4.0

   !$omp end target

   print *, x(1:10)
   print *, y(1:10)

end program pin_array
