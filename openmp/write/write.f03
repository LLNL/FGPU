program fmain
   implicit none

   real :: simple_arr(10)

   simple_arr = 1.0

   !$omp target enter data map(to:simple_arr)

! Test basic write
   !$omp target
   write (*,*) "foo ", simple_arr
   !$omp end target

! Test NEW_LINE intrisic.
   !$omp target
   write(*,*) "foo ", NEW_LINE('a'), simple_arr
   !$omp end target

! Test formatted write
   !$omp target
   write(*,100) "foo ", simple_arr

   100    format(3a,F10.2)
   !$omp end target

!$omp target exit data map(release:simple_arr)

end program fmain
