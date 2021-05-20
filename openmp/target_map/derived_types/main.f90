! make clean ; make ; nvprof -f -o test.nvvp ./test
FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  real(kind=8),  intent(in) :: x
  real(kind=8),  intent(in) :: gold
  REAL,  intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal



program derived_type_openmp
	use operation_def, only : multiplication
	use setup, only : setup_types, remove_types, op_ptr, op_ptr_b
	implicit none
	real(kind=8), dimension(:,:), pointer :: v, dv
	integer :: a, i, iop
	character(len=32) :: arg
	logical:: almost_equal

	! Size of arrays to allocate
  a = 100
  ! Operation value to use
  iop = 2
	
	! Setup derived types
	call setup_types(a,iop)

	! Allocate arrays to operate on
	allocate( v(a,a), dv(a,a) )
	
	! Initialize v
	v = 4
	
	print *, 'v = ', v(1,1)
	print *, 'array = ',op_ptr%array(1,1)
	print *, 'type-bound array = ',op_ptr_b%array(1,1)

! Call multiplication operator
print *, 'Calling operations....'

!===============================================================!

!== Direct call to multiply ====================================!
	! Perform operation on v to get dv
	!$omp target data map(to:v) map(from:dv)
	do i=1,10
		call multiplication(op_ptr, v, dv, a)
	end do
	!$omp end target data
	
  IF (.NOT.almost_equal(dv(1,1), 8.0D0,0.1)) THEN
    WRITE(*,*)  '1/ Wrong value for dv = v*array', dv(1,1)
    STOP 112
  ENDIF

! Reinit value
dv(:,:) = 0

!===============================================================!
!== Type-bound multiply procedure ==============================!
	! Perform operation on v to get dv
	!$omp target data map(to:v) map(from:dv)
	do i=1,10
		call op_ptr_b%multiply(v, dv, a)
	end do
	!$omp end target data
	
 IF (.NOT.almost_equal(dv(1,1), 8.0D0,0.1)) THEN
    WRITE(*,*)  '2/ Wrong value for dv = v*array', dv(1,1)
    STOP 112
  ENDIF

	! Deallocate arrays and derived types
	deallocate(v,dv)
	call remove_types()

end program derived_type_openmp
