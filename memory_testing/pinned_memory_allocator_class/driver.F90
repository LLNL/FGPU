program main
   use memory_allocators_mod
   use iso_c_binding
   implicit none
   
   integer, parameter ::  adqt = selected_real_kind(13, 307)

   real(kind=adqt), pointer, contiguous :: A1(:)
   real(kind=adqt), pointer, contiguous :: A2(:,:)
   real(kind=adqt), pointer, contiguous :: A3(:,:,:)

   integer, pointer, contiguous :: B1(:)
   integer, pointer, contiguous :: B2(:,:)
   integer, pointer, contiguous :: B3(:,:,:)
   
   call allocator%allocate(A1,[2])
   call allocator%allocate(A2,[2,2])
   call allocator%allocate(A3,[2,2,2])

   call allocator%allocate(B1,[2])
   call allocator%allocate(B2,[2,2])
   call allocator%allocate(B3,[2,2,2])

   print *, "After allocation..."
   print *, "associated(A1): ", associated(A1)
   print *, "associated(A2): ", associated(A2)
   print *, "associated(A3): ", associated(A3)

   print *, "associated(B1): ", associated(B1)
   print *, "associated(B2): ", associated(B2)
   print *, "associated(B3): ", associated(B3)

! IBM supports RANK built-in, but PGI apparently does not.
!   print *, "rank(A1) = ", rank(A1)
!   print *, "rank(A2) = ", rank(A2)
!   print *, "rank(A3) = ", rank(A3)
!   print *, "rank(B1) = ", rank(B1)
!   print *, "rank(B2) = ", rank(B2)
!   print *, "rank(B3) = ", rank(B3)

   print *, "shape(A1) = ", shape(A1)
   print *, "shape(A2) = ", shape(A2)
   print *, "shape(A3) = ", shape(A3)
   print *, "shape(B1) = ", shape(B1)
   print *, "shape(B2) = ", shape(B2)
   print *, "shape(B3) = ", shape(B3)

   print *, "size(A1) = ", size(A1)
   print *, "size(A2) = ", size(A2)
   print *, "size(A3) = ", size(A3)
   print *, "size(B1) = ", size(B1)
   print *, "size(B2) = ", size(B2)
   print *, "size(B3) = ", size(B3)

   call allocator%deallocate(A1)
   call allocator%deallocate(A2)
   call allocator%deallocate(A3)
   call allocator%deallocate(B1)
   call allocator%deallocate(B2)
   call allocator%deallocate(B3)

   print *, "After deallocation..."
   print *, "associated(A1): ", associated(A1)
   print *, "associated(A2): ", associated(A2)
   print *, "associated(A3): ", associated(A3)
   print *, "associated(B1): ", associated(B1)
   print *, "associated(B2): ", associated(B2)
   print *, "associated(B3): ", associated(B3)
end program 
