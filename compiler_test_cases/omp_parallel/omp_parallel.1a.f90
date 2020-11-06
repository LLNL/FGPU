! Reproducer for subscript out of bounds error.

program fmain

   implicit none
 
   integer :: n

   n = 8

   !$omp target
   !$omp teams num_teams(1) thread_limit(n)
   call foo(n)
   !$omp end teams
   !$omp end target
  
end program fmain

subroutine foo(n)
   use omp_lib
   implicit none

   integer, intent(in) :: n

   real :: a(n) ! doesn't work
   real :: b(n) ! doesn't work
!   real :: a(16) ! works
!   real :: b(16) ! works
   integer :: threadid, i
 
   !$omp declare target

   write(*,*) "n: ", n  

   a(:) = 1.0
   b(:) = 1.0

   !$omp distribute parallel do private(i) num_threads(16)
   do i=1,n
      a(i) = a(i) + b(i)
   end do

   write(*,*) "a(:) ", a(:)
   return

end subroutine foo
