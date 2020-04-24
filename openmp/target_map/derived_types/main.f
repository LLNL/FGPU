! make clean ; make ; nvprof -f -o test.nvvp ./test
program derived_type_openmp
	use operation_def, only : multiplication
	use setup, only : setup_types, remove_types, op_ptr, op_ptr_b
	
  implicit none
	real(kind=8), dimension(:,:), pointer :: v, dv
	integer :: a, i, iop
	character(len=32) :: arg

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
	
	print *, 'dv = v*array = ',dv(1,1)
	

!===============================================================!
!== Type-bound multiply procedure ==============================!
	! Perform operation on v to get dv
	!$omp target data map(to:v) map(from:dv)
	do i=1,10
		call op_ptr_b%multiply(v, dv, a)
	end do
	!$omp end target data
	
	print *, 'dv = v*array = ',dv(1,1)

	! Deallocate arrays and derived types
	deallocate(v,dv)
	call remove_types()

end program derived_type_openmp
