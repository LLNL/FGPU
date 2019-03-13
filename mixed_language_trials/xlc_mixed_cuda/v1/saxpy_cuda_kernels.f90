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

module saxpy_cuda_kernels_mod
contains

attributes(global) subroutine saxpy_gpu_cuda_kernel(a, x, y)
   implicit none 
   real :: x(:), y(:)
   real, value :: a
   integer :: i, n

   n = size(x)
   i = blockDim%x * (blockIdx%x - 1) + threadIdx%x

   if (i<=n) then
      y(i) = a*x(i)+y(i)
   end if

end subroutine saxpy_gpu_cuda_kernel

end module saxpy_cuda_kernels_mod
