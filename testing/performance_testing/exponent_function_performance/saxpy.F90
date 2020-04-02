!! Convert
!! Need at least four warps to fill instructions issue pipeline

module mathOps
!  integer, parameter, public :: adqt = selected_real_kind(13, 307)
  integer, parameter, public :: adqt = 8
!  integer, parameter, public :: adqt = 4

contains

  attributes(global) subroutine test_exp(a)
    implicit none
    real(adqt) :: a(1)
    real(adqt) :: b, c
    integer :: n

    do n=1,1024
      c = 1/n
      b = exp(a(1) * c)
    end do

    a(1) = b

  end subroutine test_exp
end module mathOps

program fmain
  use mathOps
  use cudafor
  implicit none
  real(adqt) :: a(1)
  real(adqt), device :: a_d(1)
  
  a(1) = 3.14159265359
  a_d = a
  call test_exp<<<56, 1024>>>(a_d)

end program fmain

