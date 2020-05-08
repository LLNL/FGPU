! Example of invalid map to GPU.  Used to test with OpenMP runtime handling of errors.

subroutine sub2_f
  implicit none
  integer a(10)
  integer i, j

  i = 1
  j = -1

  !$omp target map(to: i, j, a(i:j))
  print *, a(i:j)
  !$omp end target

  print *, 'bye'
end subroutine sub2_f

subroutine sub1_f
  call sub2_f
end subroutine sub1_f

#if !defined(DISABLE_FORTRAN_MAIN)
program main
  call sub1_f
end
#endif
