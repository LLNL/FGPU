module example_types
   use iso_c_binding

  type, public :: typeS
    real(C_DOUBLE)                        :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
  end type typeS

  type, public :: typeG
    real(C_DOUBLE)                        :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
  end type typeG

  type, public :: typeQ
    real(C_DOUBLE)                        :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
    type(typeS), pointer, dimension(:)    :: s_array
    type(typeG), pointer, dimension(:)    :: g_array
  end type typeQ

  type(typeQ), pointer, public :: typeQ_ptr

  contains

    subroutine initialize()
      implicit none
      integer :: n

      allocate(typeQ_ptr)
      allocate(typeQ_ptr%double_array(10))

      allocate(typeQ_ptr%s_array(2))
      allocate(typeQ_ptr%g_array(2))

      do n=1,2
         allocate(typeQ_ptr%s_array(n)%double_array(5))
         allocate(typeQ_ptr%g_array(n)%double_array(5))
      enddo

    end subroutine initialize

end module example_types
