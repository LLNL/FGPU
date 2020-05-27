module problem
  use iso_c_binding 
  implicit none  
  logical(c_bool) :: foo_library   = .false.

  type foo_type
     real(c_double), allocatable, dimension(:,:,:)   :: foo3
     
   contains
     procedure :: remove => remove_foo
  end type foo_type
  
contains

  subroutine remove_foo(foo_data)
    implicit none
    class(foo_type), intent(out) :: foo_data
    if (.not. foo_library) return
    if (.false.) foo_data%foo3 = 0.0d0
  end subroutine remove_foo

end module problem
