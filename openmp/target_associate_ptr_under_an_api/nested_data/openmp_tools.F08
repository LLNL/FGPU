module openmp_tools

   use iso_c_binding
   use omp_lib
   implicit none

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr

   interface

      type(C_PTR) function omp_target_alloc( num_bytes, device_num ) bind ( c, name = 'omp_target_alloc' )
        use iso_c_binding
        implicit none

        integer(C_SIZE_T), value :: num_bytes
        integer(C_INT), value :: device_num
      end function omp_target_alloc

      subroutine omp_target_free( h_ptr, device_num ) bind ( c, name = 'omp_target_free' )
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr
        integer(C_INT), value :: device_num
      end subroutine omp_target_free

      integer (C_INT) function omp_target_associate_ptr( h_ptr, d_ptr, num_bytes, offset, device_num)
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr, d_ptr
        integer(C_SIZE_T), value :: num_bytes, offset
        integer(C_INT), value :: device_num
      end function omp_target_associate_ptr

      integer (C_INT) function omp_target_disassociate_ptr( h_ptr, device_num)
        use iso_c_binding
        implicit none

        type(C_PTR), value :: h_ptr
        integer(C_INT), value :: device_num
      end function omp_target_disassociate_ptr

   end interface

contains


! IMPORTANT NOTE
! These routines will allocate memory on the device and associate the host c_ptr with the device c_ptr.  You still need to
! perform a omp target map after this for it to work because:
! #1 The dope vector was not mapped by target_associate_ptr.
! #2 If your ptr is a derived type component, you still need an omp pragma map
!    line with the fully qualified name (ie foo%bar ) for the component to get
!    properly attached to the derived type on the device.

   subroutine target_alloc_and_associate_double_1d(h_ptr)
      use iso_c_binding
      implicit none

      real(C_DOUBLE), pointer, intent(in) :: h_ptr(:)

      integer(C_SIZE_T) :: num_bytes, offset
      integer :: err
      type(C_PTR) :: d_ptr

      num_bytes = storage_size(h_ptr,kind=C_SIZE_T)/8*SIZE(h_ptr)
      offset = 0

      ! Using omp_target_alloc as a surrogate for an external memory allocation
      ! library.
      ! This code example is meant to demonstrate use cases where using an external
      ! memory library, such as the LLNL UMPIRE library, is required.
      d_ptr = omp_target_alloc(num_bytes, omp_get_default_device() )

      err = omp_target_associate_ptr( C_LOC(h_ptr), d_ptr, num_bytes, offset, omp_get_default_device() )
      if (err /= 0) then
         print *, "Target associate failed."
      endif
      
   end subroutine target_alloc_and_associate_double_1d


   subroutine target_alloc_and_associate_double_2d(h_ptr)
      use iso_c_binding
      implicit none

      real(C_DOUBLE), pointer, intent(in) :: h_ptr(:,:)

      integer(C_SIZE_T) :: num_bytes, offset
      integer :: err
      type(C_PTR) :: d_ptr

      num_bytes = storage_size(h_ptr,kind=C_SIZE_T)/8*SIZE(h_ptr)
      offset = 0

      ! Using omp_target_alloc as a surrogate for an external memory allocation
      ! library.
      ! This code example is meant to demonstrate use cases where using an external
      ! memory library, such as the LLNL UMPIRE library, is required.
      d_ptr = omp_target_alloc(num_bytes, omp_get_default_device() )

      err = omp_target_associate_ptr( C_LOC(h_ptr), d_ptr, num_bytes, offset, omp_get_default_device() )
      if (err /= 0) then
         print *, "Target associate failed."
      endif
      
   end subroutine target_alloc_and_associate_double_2d

   subroutine target_alloc_and_associate_double_3d(h_ptr)
      use iso_c_binding
      implicit none

      real(C_DOUBLE), pointer, intent(in) :: h_ptr(:,:,:)

      integer(C_SIZE_T) :: num_bytes, offset
      integer :: err
      type(C_PTR) :: d_ptr

      num_bytes = storage_size(h_ptr,kind=C_SIZE_T)/8*SIZE(h_ptr)
      offset = 0

      ! Using omp_target_alloc as a surrogate for an external memory allocation
      ! library.
      ! This code example is meant to demonstrate use cases where using an external
      ! memory library, such as the LLNL UMPIRE library, is required.
      d_ptr = omp_target_alloc(num_bytes, omp_get_default_device() )

      err = omp_target_associate_ptr( C_LOC(h_ptr), d_ptr, num_bytes, offset, omp_get_default_device() )
      if (err /= 0) then
         print *, "Target associate failed."
      endif

   end subroutine target_alloc_and_associate_double_3d

! IMPORTANT NOTE
! These routines will free memory on the device and disassociate the host c_ptr from the device c_ptr.  You still need to
! perform a omp target map exit after this for it to work because:
! #1 The dope vector needs to be freed, that won't be handled by the target_disassociate_ptr.
! #2 If your ptr is a derived type component, you may still need an omp pragma map
!    line with the fully qualified name (ie foo%bar ) for the component to get
!    de-attached from the derived type on the device.  (MAYBE, NOT SURE ABOUT THIS...)
   subroutine target_free_and_disassociate_double_1d(h_ptr)
      use iso_c_binding
      implicit none
		real(C_DOUBLE), pointer, intent(in) :: h_ptr(:)

      type(C_PTR) :: d_ptr
      integer :: err

      d_ptr = c_null_ptr
      !$omp target data use_device_ptr(h_ptr)
      d_ptr = C_LOC(h_ptr)
      !$omp end target data
         
      if(.NOT. C_ASSOCIATED(d_ptr) ) then
        print *, "Failed to get buffer address of pointer."
        call abort
      endif

      err = omp_target_disassociate_ptr( C_LOC(h_ptr), omp_get_default_device() )
      if (err /= 0) then
         print *, "Target disassociate on x failed."
      endif
      
      call omp_target_free( d_ptr, omp_get_default_device() )

   end subroutine target_free_and_disassociate_double_1d

   subroutine target_free_and_disassociate_double_2d(h_ptr)
      use iso_c_binding
      implicit none
		real(C_DOUBLE), pointer, intent(in) :: h_ptr(:,:)

      type(C_PTR) :: d_ptr
      integer :: err

      d_ptr = c_null_ptr
      !$omp target data use_device_ptr(h_ptr)
      d_ptr = C_LOC(h_ptr)
      !$omp end target data
         
      if(.NOT. C_ASSOCIATED(d_ptr) ) then
        print *, "Failed to get buffer address of pointer."
        call abort
      endif

      err = omp_target_disassociate_ptr( C_LOC(h_ptr), omp_get_default_device() )
      if (err /= 0) then
         print *, "Target disassociate on x failed."
      endif
      
      call omp_target_free( d_ptr, omp_get_default_device() )

   end subroutine target_free_and_disassociate_double_2d

   subroutine target_free_and_disassociate_double_3d(h_ptr)
      use iso_c_binding
      implicit none
		real(C_DOUBLE), pointer, intent(in) :: h_ptr(:,:,:)

      type(C_PTR) :: d_ptr
      integer :: err

      d_ptr = c_null_ptr
      !$omp target data use_device_ptr(h_ptr)
      d_ptr = C_LOC(h_ptr)
      !$omp end target data
         
      if(.NOT. C_ASSOCIATED(d_ptr) ) then
        print *, "Failed to get buffer address of pointer."
        call abort
      endif

      err = omp_target_disassociate_ptr( C_LOC(h_ptr), omp_get_default_device() )
      if (err /= 0) then
         print *, "Target disassociate on x failed."
      endif
      
      call omp_target_free( d_ptr, omp_get_default_device() )

   end subroutine target_free_and_disassociate_double_3d

   subroutine target_alloc_and_associate_typeS_1d(h_ptr)
      use iso_c_binding
      use example_types
      implicit none

      type(typeS), pointer, intent(in) :: h_ptr(:)

      integer(C_SIZE_T) :: num_bytes, offset
      integer :: err
      type(C_PTR) :: d_ptr

      num_bytes = storage_size(h_ptr,kind=C_SIZE_T)/8*SIZE(h_ptr)
      offset = 0

      ! Using omp_target_alloc as a surrogate for an external memory allocation
      ! library.
      ! This code example is meant to demonstrate use cases where using an external
      ! memory library, such as the LLNL UMPIRE library, is required.
      d_ptr = omp_target_alloc(num_bytes, omp_get_default_device() )

      err = omp_target_associate_ptr( C_LOC(h_ptr), d_ptr, num_bytes, offset, omp_get_default_device() )
      if (err /= 0) then
         print *, "Target associate failed."
      endif

   end subroutine target_alloc_and_associate_typeS_1d

   subroutine target_free_and_disassociate_typeS_1d(h_ptr)
      use iso_c_binding
      use example_types
      implicit none
      type(typeS), pointer, intent(in) :: h_ptr(:)

      type(C_PTR) :: d_ptr
      integer :: err

      d_ptr = c_null_ptr
      !$omp target data use_device_ptr(h_ptr)
      d_ptr = C_LOC(h_ptr)
      !$omp end target data
         
      if(.NOT. C_ASSOCIATED(d_ptr) ) then
        print *, "Failed to get buffer address of pointer."
        call abort
      endif
          
      err = omp_target_disassociate_ptr( C_LOC(h_ptr), omp_get_default_device() )
      if (err /= 0) then
         print *, "Target disassociate on x failed."
      endif
      
      call omp_target_free( d_ptr, omp_get_default_device() )

   end subroutine target_free_and_disassociate_typeS_1d

   subroutine target_alloc_and_associate_typeQ_ptr(h_ptr)
      use iso_c_binding
      use example_types
      implicit none

      type(typeQ), pointer, intent(in) :: h_ptr

      integer(C_SIZE_T) :: num_bytes, offset
      integer :: err
      type(C_PTR) :: d_ptr

      num_bytes = storage_size(h_ptr,kind=C_SIZE_T)/8
      offset = 0

      ! Using omp_target_alloc as a surrogate for an external memory allocation
      ! library.
      ! This code example is meant to demonstrate use cases where using an external
      ! memory library, such as the LLNL UMPIRE library, is required.
      d_ptr = omp_target_alloc(num_bytes, omp_get_default_device() )

      err = omp_target_associate_ptr( C_LOC(h_ptr), d_ptr, num_bytes, offset, omp_get_default_device() )
      if (err /= 0) then
         print *, "Target associate failed."
      endif

   end subroutine target_alloc_and_associate_typeQ_ptr

   subroutine target_free_and_disassociate_typeQ_ptr(h_ptr)
      use iso_c_binding
      use example_types
      implicit none
      type(typeQ), pointer, intent(in) :: h_ptr

      type(C_PTR) :: d_ptr
      integer :: err

      d_ptr = c_null_ptr
      !$omp target data use_device_ptr(h_ptr)
      d_ptr = C_LOC(h_ptr)
      !$omp end target data
 
      if(.NOT. C_ASSOCIATED(d_ptr) ) then
        print *, "Failed to get buffer address of pointer."
        call abort
      endif
      
      err = omp_target_disassociate_ptr( C_LOC(h_ptr), omp_get_default_device() )
      if (err /= 0) then
         print *, "Target disassociate on x failed."
      endif
      
      call omp_target_free( d_ptr, omp_get_default_device() )

   end subroutine target_free_and_disassociate_typeQ_ptr

end module openmp_tools
