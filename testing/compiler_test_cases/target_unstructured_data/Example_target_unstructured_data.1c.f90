! Shows issue with omp pointer properties not propogating between 'self' and 'type_ptr'.
! @@name:        target-unstructured-data.1c.f90
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
    subroutine initialize_direct(N)
      implicit none
      integer :: N

      type_ptr%x = 100
      allocate(type_ptr%A(N))

      !$omp target enter data map(alloc:type_ptr)
      !$omp target enter data map(alloc:type_ptr%A)

    end subroutine initialize_direct

    subroutine finalize()
      implicit none

      !$omp target exit data map(delete:type_ptr%A)
      !$omp target exit data map(delete:type_ptr)
      deallocate(type_ptr%A)

    end subroutine finalize
end module example


program fmain
   use example

   implicit none
   
   allocate(type_ptr)
   call initialize_direct(5)

   type_ptr%x = 1.0
   type_ptr%A(:) = 1.0

   print *, "Host, before: ", type_ptr%x, type_ptr%A(:)

!$omp target update to(type_ptr, type_ptr%A)

!$omp target
   call foo()
!$omp end target 

!$omp target update from(type_ptr, type_ptr%A)

   print *, "Host, after: ", type_ptr%x, type_ptr%A(1)

   call finalize()

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

