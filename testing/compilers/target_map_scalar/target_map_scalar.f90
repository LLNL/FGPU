program main
   implicit none

   integer ::  x
   integer, dimension(1) :: y
 
   x = 0
   y(1) = 0
    
   print *, "x, before: ", x
   print *, "y, before: ", y(1)

   !$omp target data map(tofrom:x,y)

   !$omp target
   x = x + 1
   y(1) = y(1) + 1
   write (*,*) "x, device: ", x
   write (*,*) "y, device: ", y(1)
   !$omp end target

   !$omp end target data

   print *, "x, end: ", x
   print *, "y, end: ", y(1)

end program main
