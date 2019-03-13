!*****************************************************************************
!
! Different implementations of a simple single precision A-X plus Y function
! on a CPU and GPU, to verify this code works.
!
! x = ax + y
! x,y,z: vector
! a: scalar
!*****************************************************************************


!*****************************************************************************
! Kernels
!*****************************************************************************

module saxpy_openmp_kernels_mod
contains

subroutine print_thread_info
   integer :: myid, nthreads
   integer :: OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM

   !$OMP PARALLEL default(none) private(myid) &
   !$OMP shared(nthreads)

   ! Determine number of threads and their id
   myid = OMP_GET_THREAD_NUM()
   nthreads = OMP_GET_NUM_THREADS();

   !$OMP BARRIER

   if (myid == 0) then
      print *,'nthreads=', nthreads
   end if

   !$OMP END PARALLEL
end subroutine print_thread_info


!*****************************************************************************
! CPU parallel (openmp)
!*****************************************************************************
subroutine saxpy_cpu_openmp_kernel(n, a, x, y)
   real :: x(:), y(:), a
   integer :: n, i

!$omp parallel do
   do i=1,n
      y(i) = a*x(i)+y(i)
   end do
!$omp end parallel do

end subroutine saxpy_cpu_openmp_kernel


!*****************************************************************************
! GPU parallel (openmp 4.5)
!*****************************************************************************
subroutine saxpy_gpu_openmp_kernel(n, a, x, y)
   real :: x(:), y(:), a
   integer :: n, i

!$omp target
!$omp parallel do
   do i=1,n
      y(i) = a*x(i)+y(i)
   end do
!$omp end parallel do
!$omp end target

end subroutine saxpy_gpu_openmp_kernel

end module saxpy_openmp_kernels_mod
