program main
   use cudafor
   implicit none
   

!  With allocatable, pinned - 9 calls, HtoD 3.98us, 704ns, 15.us
!                             5 calls, DtoH 11.5us, 640ns, 18.9us
!       pointer, not pinned - 8 calls, HtoD 4.7us, 704ns, 19.0us
!                             4 calls, DtoH 11.3us, 640ns, 18.5us

   real, allocatable, pinned  ::  p(:), v1(:), v2(:)
   integer ::  i, N

   N = 100000

   allocate(p(N))
   allocate(v1(N))
   allocate(v2(N))

   do i=1,N
      p(i) = 0.0
      v1(i) = i*2.0
      v2(i) = i*3.0
   end do
      
   !$omp target map(to:i, N, v1, v2) map(from:p)
   !$omp parallel do
   do i=1,N
      p(i) = v1(i) * v2(i)
   end do
   !$omp end target

   print *, p(1)
end program main
