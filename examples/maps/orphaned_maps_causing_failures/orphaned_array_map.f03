subroutine foo1
	implicit none

	double precision, dimension(10,10) :: a, b

	print *, "----------- Mapping foo1 a ----------"
   !$omp target enter data map(to:a)
	print *, "----------- Mapping foo1 b ----------"
	!$omp target enter data map(to:b)

! These local variables will go out of scope, leaving behind
! two orphaned entries in the host<->device address map registry.

end subroutine foo1

subroutine foo2
	implicit none

	double precision, dimension(5,5) :: c

! This likely results in an OpenMP mapping error, because the
! CPU may put 'c' in memory previously occupied by foo1's a or b arrays.

	print *, "----------- Mapping foo1 c ----------"
	!$omp target enter data map(to:c)

end subroutine foo2

program map_testing
	implicit none
	
	call foo1
	call foo2

end program map_testing
