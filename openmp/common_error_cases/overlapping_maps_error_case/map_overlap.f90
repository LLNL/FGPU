program main
  use omp_lib
  implicit none
  integer :: a(50)
  a = -8
!$omp target data map(a(10:20))
  !$omp target map(a(15:30))
     a(15:30) = 300
  !$omp end target
  print *, "--3--", a
!$omp end target data
  print *, "--3--", a
end program
