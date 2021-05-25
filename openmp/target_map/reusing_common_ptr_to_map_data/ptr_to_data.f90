FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  DOUBLE PRECISION, intent(in) :: x
  DOUBLE PRECISION, intent(in) :: gold
  REAL,     intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal

module data
   double precision, target, dimension(10,10) :: a, b
   double precision, pointer, contiguous :: data_ptr(:,:)
end module data

program map_testing
   use data
   implicit none
   LOGICAL :: almost_equal

   a = 1.0
   b = 2.0

   data_ptr => a
   print *, "----------- Mapping a via data_ptr----------"
   !$omp target enter data map(to:data_ptr)

   data_ptr => b
   print *, "----------- Mapping b via data_ptr----------"
   !$omp target enter data map(to:data_ptr)

	!$omp target
	a(1,1) = 10.0
	b(1,1) = 20.0
	!$omp end target

	data_ptr=>a
	!$omp target exit data map(from:data_ptr)
	data_ptr=>b
	!$omp target exit data map(from:data_ptr)

  IF ( .NOT.almost_equal(a(1,1), 10.0D0, 0.1) ) THEN
    WRITE(*,*)  'Expected', 10.0,  'Got', a(1,1)
    STOP 112
  ENDIF

  IF ( .NOT.almost_equal(b(1,1), 20.0D0, 0.1) ) THEN
    WRITE(*,*)  'Expected', 20.0,  'Got', b(1,1)
    STOP 112
  ENDIF
  WRITE(*,*) "Look good"

end program map_testing
