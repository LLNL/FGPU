subroutine foo(i, a, b)
  use omp_lib
  integer a, b
  integer c
  integer :: d = 10

  !$omp barrier

  print *, "t#: ", omp_get_thread_num(), " i: ", i, " loc(a):", loc(a), "  loc(b): ", loc(b), "  loc(c): ", loc(c), "  loc(d): ", loc(d)

end subroutine foo


program test
  use omp_lib
  integer :: a = 1
  integer :: b = 5
  integer :: i

  print *, "loc(a): ", loc(a), "  loc(b): ", loc(b)

  !$omp parallel private(i)
  i = omp_get_thread_num()
  call foo(i, a, b)
  !$omp end parallel

end
