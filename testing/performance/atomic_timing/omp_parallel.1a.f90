! Time atomics vs SUM

program fmain

   use omp_lib
   implicit none
 
   integer :: n(128000), exited(128000)
   integer :: not_done(128000)
   integer :: i,len, num_left
   double precision :: ostart, oend

   len = 128000
   n = 1
   exited = 0
   not_done = 1

   ostart = omp_get_wtime()
   IterationLoop: do

      num_left = 0

      !omp parallel do private(i)
      do i=1,len

         if (exited(i) .eq. 0) then
            n(i) = ( n(i) + i ) * 2

            if (n(i) > len ) then
               exited(i) = 1
!            else
!      !$omp atomic
!               num_left = num_left + 1
!      !$omp end atomic
            endif
         endif
      enddo

!      if (num_left .eq. 0) then
!         exit IterationLoop
!      else
!         print *,"Num left: ", num_left
!      endif

      num_left = SUM(exited)
      if (num_left .eq. len) then
         exit IterationLoop
      else
         print *,"Num left: ", num_left
      endif
   enddo IterationLoop
   
   oend = omp_get_wtime()
   print *,"OpenMP Walltime: ", oend-ostart

end program fmain
