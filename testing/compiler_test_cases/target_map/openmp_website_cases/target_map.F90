program main
   implicit none

   type :: derivedtype
      integer :: foo
      real, pointer  ::  p(:), v1(:), v2(:)
   end type derivedtype

   type(derivedtype) :: dt
   integer ::  i, N

   N = 100
   allocate(dt%p(100))
   allocate(dt%v1(100))
   allocate(dt%v2(100))

   dt%foo = 1

   do i=1,N
      dt%p(i) = 0.0
      dt%v1(i) = i*2.0
      dt%v2(i) = i*3.0
   end do
      
   !$omp target data map(to:dt)
   !$omp data map(to:dt%v1, dt%v2) map(from:dt%p)
   !$omp parallel do
   do i=1,N
      print *, dt%foo
      dt%p(i) = dt%v1(i) * dt%v2(i)
   end do
   !$omp end target

end program main
