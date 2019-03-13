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

   !$omp parallel private(i, threadid) num_threads(16)
   write(*,*) "Num threads: ", omp_get_num_threads()
   threadid = omp_get_thread_num()
   write(*,*) "Thread id: ", threadid

!  Change to 1..n index for fortran array
   i = threadid + 1

   if (i <= n) then
      write(*,*) "a(",i,") access"
      a(i) = a(i) + b(i)
   else
      write(*,*) "Thread id: ", threadid, ", > 16?!"
   endif
   !$omp end parallel

   return

end subroutine foo
