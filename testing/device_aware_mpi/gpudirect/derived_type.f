module derived_type
	implicit none

	type value_type
		real(kind=8), dimension(:,:), pointer :: array
		real(kind=8), dimension(:,:), pointer :: send,recv
		real(kind=8) :: scalar
		integer :: nx, ny				! Array dimensions
	end type value_type
	
	type operation_type
		integer :: null_op
		type(value_type), dimension(2) :: mult, add
	end type operation_type
	
	contains
	
	subroutine setup_values(op,nx,ny,i)
		implicit none
		type(value_type) :: op		! derived type holding values for performing operations
		integer :: nx, ny, i		! size and value of arrays
		
		! Allocate and assign values to components
		allocate( op%array(nx,ny), op%send(nx,ny), op%recv(nx,ny))
		op%array = i
		op%scalar = i
		op%nx = nx; op%ny = ny

	end subroutine setup_values
	
	subroutine remove_values(op)
		type(value_type), intent(out) :: op
		continue
	end subroutine remove_values
	
end module derived_type
