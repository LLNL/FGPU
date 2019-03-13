! Time atomics vs SUM

program fmain

   use omp_lib
   implicit none
 
   integer :: n(128000), exited(128000)
   double precision :: x(128000), y(128000), z(128000)
   integer :: not_done(128000)
   integer :: i,len, num_left
   double precision :: ostart, oend

   len = 128000
   n = 1
   exited = 0
   not_done = 1

   y = 10.0

   ostart = omp_get_wtime()
!$omp parallel do schedule(static)
   do i=1,len
      x(i) = y(i)
      z(i) = y(i)
   enddo
   oend = omp_get_wtime()
   print *,"OpenMP Walltime: ", oend-ostart

end program fmain
