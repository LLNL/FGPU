program main
   use cudafor
   implicit none
   

!  With allocatable, pinned - 8 calls, HtoD 852ns, 704ns, 1.6960us
!                             4 calls, DtoH 648ns, 640ns, 672ns
!       pointer, not pinned - 8 calls, HtoD 732ns, 704ns, 768ns
!                             4 calls, DtoH 616ns, 608ns, 640ns

   type :: derivedtype
      real, allocatable, pinned  ::  p(:), v1(:), v2(:)
   end type derivedtype

   type(derivedtype) :: dt
   integer ::  i, N

   N = 100
   allocate(dt%p(100))
   allocate(dt%v1(100))
   allocate(dt%v2(100))

   do i=1,N
      dt%p(i) = 0.0
      dt%v1(i) = i*2.0
      dt%v2(i) = i*3.0
   end do
      
   !$omp target map(dt%v1, dt%v2) map(from:dt%p)
   !$omp parallel do
   do i=1,N
      dt%p(i) = dt%v1(i) * dt%v2(i)
   end do
   !$omp end target

   print *, dt%p(1)
end program main
