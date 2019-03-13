! Shows issue with omp target data map not copying over type_ptr%x
! type_ptr%A maps over fine.
! @@name:        target-unstructured-data.1.f
! @@type:        F-free
! @@compilable:  yes
! @@linkable:    no
! @@expect:      success
module example

  type, public :: example_type
    real(8)              :: x
    real(8), pointer     :: A(:)
    
  end type example_type

  !$omp declare target (type_ptr)
  type(example_type), pointer, public :: type_ptr

  contains
    subroutine initialize(self,N)
      implicit none
      type(example_type), pointer, intent(inout) :: self
      integer :: N

      self%x = 100
      allocate(self%A(N))

    end subroutine initialize

    subroutine finalize(self)
      implicit none
      type(example_type), pointer, intent(inout) :: self

      deallocate(self%A)

    end subroutine finalize
end module example


program fmain
   use example

   implicit none
   
   allocate(type_ptr)
   call initialize(type_ptr,5)

   type_ptr%x = 1.0
   type_ptr%A(:) = 1.0

   print *, "Host, before: ", type_ptr%x, type_ptr%A(:)

!$omp target data map(tofrom:type_ptr)
!$omp target data map(tofrom:type_ptr%A)

!!$omp target map(tofrom:type_ptr, type_ptr%A)
   call foo()
!!$omp end target 

!$omp end target data
!$omp end target data

   print *, "Host, after: ", type_ptr%x, type_ptr%A(1)

   call finalize(type_ptr)

   deallocate(type_ptr)

end program fmain

subroutine foo()
   use example
   implicit none

   !$omp declare target

   write (*,*) "Device, before:", type_ptr%x, type_ptr%A(1)

   type_ptr%x = 2.0
   type_ptr%A(1) = 2.0

   write (*,*) "Device, after: ", type_ptr%x, type_ptr%A(1)
   return

end subroutine

