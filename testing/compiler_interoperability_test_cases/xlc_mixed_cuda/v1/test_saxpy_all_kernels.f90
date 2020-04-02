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
! Setup/driver code
!*****************************************************************************

subroutine test_saxpy_openmp_kernels()
   use saxpy_openmp_kernels_mod
   use saxpy_kernels_mod

   ! Number of elements in arrays
   integer :: n 
   ! Arrays
   real :: x(2**20), y(2**20)
   ! Scalar value to multiply array x by.
   real :: a

   n = 2**20
   a = 2.0
   ! Initialize arrays
   x = 1.0
   y = 2.0

   ! Reference calculation - serial cpu code.
   do i=1,n
      y(i) = a*x(i)+y(i)
   end do

   ! OpenMP CPU
   y = 2.0
   call saxpy_cpu_openmp_kernel(n, a, x, y)

   ! OpenMP 4.5 GPU
   y = 2.0
   call saxpy_gpu_openmp_kernel(n, a, x, y)

end subroutine test_saxpy_openmp_kernels

subroutine test_saxpy_cuda_kernels
  use cudafor
  use saxy_cuda_kernels_mod

  real, device :: x_d(2**20), y_d(2**20)
  x_d = 1.0
  y_d = 2.0

  ! Perform SAXPY on 1M elements
  call saxpy_gpu_cuda_kernel<<<4096, 256>>>(2.0, x_d, y_d)

  return
end subroutine test_saxpy_cuda_kernels

