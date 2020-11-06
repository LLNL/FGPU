program main
   use cudafor
   use iso_c_binding
   implicit none

   type :: derivedtype
      integer :: n1, n2, n3
      real, pointer   ::  v1(:), v2(:,:), v3(:,:,:)
   end type derivedtype

   integer :: i, j, k, err
   type(derivedtype) :: dt
   type(C_PTR) :: cptr

   err = cudaHostAlloc( cptr, 10*sizeof(0.0), cudaHostAllocPortable )
   print *, err
   call c_f_pointer( cptr,dt%v1, [10] ) 

   err = cudaHostAlloc( cptr, 10*10*sizeof(0.0), cudaHostAllocPortable )
   print *, err
   call c_f_pointer( cptr,dt%v2,[10,10] )

   err = cudaHostAlloc( cptr, 10*10*10*sizeof(0.0), cudaHostAllocPortable )
   print *, err
   call c_f_pointer( cptr,dt%v3, [10,10,10] )

!   allocate(dt%v1(10))
!   allocate(dt%v2(10,10))
!   allocate(dt%v3(10,10,10))

! Allocate memory on gpu early.
   !$omp target data map(alloc: dt)
   !$omp target data map(alloc: dt%v1, dt%v2, dt%v3)

! Mimick CPU setup of values

   dt%n1 = 10
   dt%n2 = 10
   dt%n3 = 10

   dt%v1(:) = 1.0
   dt%v2(:,:) = 2.0
   dt%v3(:,:,:) = 3.0
   
! Populate values on GPU in advance
   !$omp target update to(dt)
   !$omp target update to(dt%n1, dt%n2, dt%n3, dt%v1,dt%v2,dt%v3)

! Do some CPU work...
call system('sleep 1')

! Do work on GPU
!! This doesn't update dt%v2 on the host, unless update is added on line 53.
   !$omp target map(from: dt%v2)
   !$omp teams num_teams(dt%n1) thread_limit(dt%n2)
   !$omp distribute parallel do collapse(2)
   do i=1,dt%n1
      do j=1, dt%n2
         do k=1, dt%n3
            dt%v2(j,i) = dt%v2(j,i) + dt%v1(k) * dt%v3(j,i,k)
         end do
      end do
   end do
   !$omp end distribute parallel do
   !$omp end teams
   !$omp end target

   !$omp target update from(dt%v2)

   !$omp end target data
   !$omp end target data

print *, "dt%v2(1,1): ", dt%v2(1,1)
end program main
