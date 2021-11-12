module data
   implicit none

   double precision, pointer, contiguous, dimension(:,:) :: a

contains

   subroutine map_to(ptr)
      double precision, intent(in), pointer, contiguous, dimension(:,:) :: ptr
      double precision, pointer, contiguous, dimension(:,:) :: ptr_alias

      ptr_alias => ptr
      !$omp target enter data map(to:ptr_alias)
   end subroutine map_to

   subroutine map_delete(ptr)
      double precision, intent(in), pointer, contiguous, dimension(:,:) :: ptr
      double precision, pointer, contiguous, dimension(:,:) :: ptr_alias
      ptr_alias => ptr
      !$omp target exit data map(delete:ptr_alias)
   end subroutine map_delete

end module data


program map_testing
   use data
   implicit none

   integer i

   allocate(a(10,10))

   do i = 1, 100
      print *, "----------- Mapping a, iteration ", i, "----------"
      call map_to(a)
   enddo

   do i = 1, 100
      print *, "----------- Unmapping a, iteration ", i, "----------"
      call map_delete(a)
   enddo

end program map_testing
