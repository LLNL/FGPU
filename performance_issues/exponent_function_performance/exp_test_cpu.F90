program fmain

  implicit none
  include 'mpif.h'

  real(8) :: a(1)
  real(8) :: b, c
  integer :: n, i_error
  real(8) :: t1, t2

  call MPI_INIT(i_error)

  a(1) = 3.14159265359

  t1 = MPI_Wtime()

  do n=1,10000000
    c = 1/n
    b= exp(a(1) * c)
  end do

  a(1) = b

  t2 = MPI_Wtime()

  print *, t2 - t1
end program fmain
