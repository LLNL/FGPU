module operation_def
	use base_types, only : base_type
	use omp_lib
	
	implicit none
	
	! Define custom operator types
	
	! Type-bound procedure
	type, extends(base_type) :: multiply_type
	  integer :: null_op
		contains
		procedure :: multiply => multiplication_b
	end type

	contains
	
	subroutine multiplication(op,v,dv,n)
		type(base_type) :: op
		real(kind=8), dimension(op%nx,op%ny), intent(in) :: v
		real(kind=8), dimension(op%nx,op%ny), intent(out) :: dv
		integer i,j,n

		!$omp target teams distribute parallel do collapse(2)
		do i=1,op%nx
			do j=1,op%ny
				dv(i,j) = op%array(i,j)*v(i,j)
			end do ! j
		end do ! i
		!$omp end target teams distribute parallel do
		
	end subroutine multiplication
	
	subroutine multiplication_b(op,v,dv,n)
		class(multiply_type) :: op
		real(kind=8), dimension(op%nx,op%ny), intent(in) :: v
		real(kind=8), dimension(op%nx,op%ny), intent(out) :: dv
		integer i,j,n

		!$omp target teams distribute parallel do collapse(2)
		do i=1,op%nx
			do j=1,op%ny
				dv(i,j) = op%array(i,j)*v(i,j)
			end do ! j
		end do ! i
		!$omp end target teams distribute parallel do
		
	end subroutine multiplication_b
	
end module operation_def
