module data
   double precision, target, dimension(10,10) :: a, b
end module data

subroutine foo1
   use data
   implicit none

   double precision, pointer, contiguous :: data_ptr(:,:)

! This will map two items.
! - the FORTRAN 'dope' pointer object (shape, offset, address of data)
! - the data contents pointed to by the FORTRAN pointer object
   data_ptr => a
   print *, "----------- Mapping a via data_ptr----------"
   !$omp target enter data map(to:data_ptr)

! data_ptr will now go out of scope.  The memory uses by the FORTRAN
! dope pointer object is reclaimed by CPU.  You now have an orphaned
! entry in the OpenMP host<->device address registry if you don't
! remove that mapping.

end subroutine foo1

subroutine foo2
   implicit none

   double precision, dimension(5,5) :: c

! This will generate an 'overlapping data map' error because the
! address of 'c' likely overlaps the address previously used by
! foo1's data_ptr dope pointer.

   print *, "----------- Mapping foo1 c ----------"
   !$omp target enter data map(to:c)

end subroutine foo2

program map_testing
   implicit none

   call foo1
   call foo2

end program map_testing
