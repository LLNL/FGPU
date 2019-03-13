! @@name:        target-unstructured-data.1.f
! @@type:        F-free
! @@compilable:  yes
! @@linkable:    no
! @@expect:      success
module example

! ADDED OMP DECLARE - Missing from online example.
  !$omp declare target (A)
  real(8), allocatable :: A(:)

  contains
    subroutine initialize(N)
      integer :: N

      allocate(A(N))
      !$omp target enter data map(alloc:A)

    end subroutine initialize

    subroutine finalize()

      !$omp target exit data map(delete:A)
      deallocate(A)

    end subroutine finalize
end module example


program fmain
   use example

   implicit none
   
   call initialize(5)

   A(:) = 1.0

   print *, "Host, before: ", A(:)

!$omp target update to(A)

!$omp target map(A)
   call foo()
!$omp end target

!$omp target update from(A)

   print *, "Host, after: ", A(:)

   call finalize()
end program fmain

subroutine foo()
   use example
   implicit none

   !$omp declare target

   ! This write doesn't work.
   write (*,*) "Device, before:", A(:)

   A(:) = 2.0
   A(1) = 2.0

   ! This write doesn't work.
   write (*,*) "Device, after: ", A(:)
   ! This one works.
   write (*,*) "Device, after: ", A(1)
   return

end subroutine

