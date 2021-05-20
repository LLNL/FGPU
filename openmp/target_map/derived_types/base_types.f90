module base_types
  implicit none

! Define base derived types

!== Type-bound procedure ========================! 
	type base_type
		real(kind=8), dimension(:,:), pointer :: array
		real(kind=8) :: scalar
		integer :: nx, ny				! Array dimensions
		integer :: openmp				! Flag for different openmp implementations
		contains
		procedure :: setup => setup_values
	end type base_type

	contains
	
!== Type-bound procedures =========================!
	subroutine setup_values(op,nx,ny,i)
		implicit none
		class(base_type) :: op		! derived type holding values for performing operations
		integer :: nx, ny, i		! size of arrays
		
		! Allocate and assign values to components
		allocate( op%array(nx,ny))
		op%array = i
		op%scalar = i
		op%nx = nx; op%ny = ny

	end subroutine setup_values

end module base_types
