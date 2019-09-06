module setup
	use derived_type, only : operation_type, setup_values, remove_values
	implicit none
	type(operation_type), target :: operation
	type(operation_type), pointer :: op_ptr
	
	contains
	
	subroutine setup_types(a,val)
		implicit none
		integer :: i, a, val
		
		do i=1,2
			! Setup multiplication operation
			call setup_values(operation%mult(i),a,a,val)
			! Setup addition operation
			call setup_values(operation%add(i),a,a,val)
		end do
		! Set pointer
		op_ptr => operation
		
		! Copy derived type array to gpu
		!do i=1,2				! Using do loop for enter data map gives ambiguous map runtime error
		 !$omp target enter data map(to:op_ptr%mult(1)%array) map(alloc:op_ptr%mult(1)%send,op_ptr%mult(1)%recv)
		!end do

	end subroutine setup_types
	
	subroutine remove_types()
		implicit none
		integer :: i
		do i=1,2
			call remove_values(operation%mult(i))
			call remove_values(operation%add(i) )
			
		end do
		!$omp target exit data map(from:op_ptr%mult(1)%array) map(delete:op_ptr%mult(1)%send,op_ptr%mult(1)%recv)
	end subroutine remove_types
	
end module setup
