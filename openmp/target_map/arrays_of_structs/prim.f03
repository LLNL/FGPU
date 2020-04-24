module prim
  use, intrinsic :: iso_c_binding
  integer :: nx = 10
  integer :: ny = 10
  integer :: nz = 10
  implicit none
  
  type prim_type
     real(c_double), allocatable, dimension(:,:,:) :: v1
     real(c_double), allocatable, dimension(:,:,:) :: v2
     real(c_double), allocatable, dimension(:,:,:) :: v3
   contains 
     procedure :: setup => setup_prim
  end type prim_type
  
    
contains
  subroutine setup_prim(prim_data)
    class(prim_type), intent(OUT) :: prim_data
    
    allocate(prim_data%v1(nx,ny,nz))
    allocate(prim_data%v2(nx,ny,nz))
    allocate(prim_data%v3(nx,ny,nz))
    
  end subroutine setup_prim
  

end module prim
