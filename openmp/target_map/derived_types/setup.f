module setup
	use base_types, only : base_type, setup_values, remove_values
	use operation_def, only : multiply_type
	implicit none
	type(base_type), target :: operation   ! Derived type without multiply operation bound
	type(base_type), pointer :: op_ptr
	type(multiply_type), target :: operation_b   ! Type-bound procedure included
	type(multiply_type), pointer :: op_ptr_b
	
	contains
	
	subroutine setup_types(a,i)
		implicit none
		integer :: i, a
	  ! Setup multiplication operation
	  call operation_b%setup(a,a,i)
	  call operation%setup(a,a,i)
	  
		! Set pointer
		op_ptr_b => operation_b
		op_ptr => operation
		
		! Map derived type data to GPU
		!$omp target enter data map(to:op_ptr)
		!$omp target enter data map(to:op_ptr_b)
		
	end subroutine setup_types
	
	subroutine remove_types()
		implicit none
		integer :: i
	  call operation_b%remove()
	  call operation%remove()
	  
	  ! Remove derived type data from GPU
		!$omp target exit data map(delete:op_ptr)
		!$omp target exit data map(delete:op_ptr_b)

	end subroutine remove_types
	
end module setup
