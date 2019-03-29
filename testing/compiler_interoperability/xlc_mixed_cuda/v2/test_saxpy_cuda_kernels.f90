!*****************************************************************************
! Setup/driver code for CUDA kernel
!*****************************************************************************
subroutine test_saxpy_cuda_kernels
  use cudafor
  use saxpy_cuda_kernels_mod
  implicit none

  real, device :: x_d(2**20), y_d(2**20)
  x_d = 1.0
  y_d = 2.0

  ! Perform SAXPY on 1M elements
  call saxpy_gpu_cuda_kernel<<<4096, 256>>>(2.0, x_d, y_d)

  return
end subroutine test_saxpy_cuda_kernels

