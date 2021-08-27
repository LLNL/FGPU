FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  REAL, intent(in) :: x
  INTEGER,  intent(in) :: gold
  REAL,     intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal


subroutine testsaxpy_omp45_f

  use iso_c_binding
  use omp_lib

  implicit none
  integer, parameter :: N = ishft(1,21)
  integer :: i, err
 
  real, pointer :: x(:), y(:)
  type(c_ptr) :: x_cptr, y_cptr
  integer(c_size_t) :: num_bytes, offset
  real :: a
  LOGICAL :: almost_equal

  allocate( x(N), y(N) )
  x = 1.0
  y = 2.0
  a = 2.0

  num_bytes = sizeof(a)*N
  offset = 0

  x_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  err = omp_target_associate_ptr( C_LOC(x), x_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on x failed."
  endif

  y_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  err = omp_target_associate_ptr( C_LOC(y), y_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on y failed."
  endif

  !$omp target update to(x,y)

  !$omp target teams distribute parallel do private(i) shared(y,a,x) default(none)
  do i=1,N
    y(i) = a*x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do

  !$omp target update from(y)

  err = omp_target_disassociate_ptr( C_LOC(x), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on x failed."
  endif
  call omp_target_free( x_cptr, omp_get_default_device() )

  err = omp_target_disassociate_ptr( C_LOC(y), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on y failed."
  endif
  call omp_target_free( y_cptr, omp_get_default_device() )

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))

  IF ( .NOT.almost_equal( maxval(abs(y-4.0)) , 0, 0.01) ) THEN
    STOP 112
  ENDIF

end subroutine testsaxpy_omp45_f

program hello
  call testsaxpy_omp45_f
end program hello

