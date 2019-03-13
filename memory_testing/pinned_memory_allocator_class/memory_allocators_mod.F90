module memory_allocators_mod

   use iso_c_binding, only : c_int, c_double, c_f_pointer, c_ptr, c_loc
#if defined (USE_CUDA)
   use cudafor, only : cudaHostAlloc, cudaHostAllocDefault, cudaSuccess, cudaFreeHost
#endif
   implicit none
   private

   type, public :: NativeAllocator
   contains
      private
      procedure :: nar1 => native_allocate_real_1
      procedure :: nar2 => native_allocate_real_2
      procedure :: nar3 => native_allocate_real_3
      procedure :: nai1 => native_allocate_int_1
      procedure :: nai2 => native_allocate_int_2
      procedure :: nai3 => native_allocate_int_3
      procedure :: ndr1 => native_deallocate_real_1
      procedure :: ndr2 => native_deallocate_real_2
      procedure :: ndr3 => native_deallocate_real_3
      procedure :: ndi1 => native_deallocate_int_1
      procedure :: ndi2 => native_deallocate_int_2
      procedure :: ndi3 => native_deallocate_int_3
      generic, public :: allocate => nar1, nar2, nar3, nai1, nai2, nai3
      generic, public :: deallocate => ndr1, ndr2, ndr3, ndi1, ndi2, ndi3

   end type NativeAllocator
#if defined (USE_CUDA)
   type, public :: PinnedAllocator
   contains
      private
      procedure :: par1 => pinned_allocate_real_1
      procedure :: par2 => pinned_allocate_real_2
      procedure :: par3 => pinned_allocate_real_3
      procedure :: pai1 => pinned_allocate_int_1
      procedure :: pai2 => pinned_allocate_int_2
      procedure :: pai3 => pinned_allocate_int_3
      procedure :: pdr1 => pinned_deallocate_real_1
      procedure :: pdr2 => pinned_deallocate_real_2
      procedure :: pdr3 => pinned_deallocate_real_3
      procedure :: pdi1 => pinned_deallocate_int_1
      procedure :: pdi2 => pinned_deallocate_int_2
      procedure :: pdi3 => pinned_deallocate_int_3
      generic, public :: allocate => par1, par2, par3, pai1, pai2, pai3
      generic, public :: deallocate => pdr1, pdr2, pdr3, pdi1, pdi2, pdi3

   end type PinnedAllocator
#endif

!-----------------------------------------------------------------------------------------
! Public allocator instance.
! TODO - Could convert this to a runtime selection.  Create a parent Allocator class,
! and extend it to Native and Pinned allocators.  Create the allocator at problem init.
!-----------------------------------------------------------------------------------------
#if defined(USE_CUDA)
   type(PinnedAllocator), public :: allocator
#else
   type(NativeAllocator), public :: allocator
#endif

!-----------------------------------------------------------------------------------------
! FORTRAN native allocator, deallocator
!-----------------------------------------------------------------------------------------
contains

! REAL data type
   subroutine native_allocate_real_1(this, ptr, dims)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1)) )
      return
   end subroutine native_allocate_real_1

   subroutine native_allocate_real_2(this, ptr, dims)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1), dims(2)) )
      return
   end subroutine native_allocate_real_2

   subroutine native_allocate_real_3(this, ptr, dims)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:,:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1), dims(2), dims(3)) )
      return
   end subroutine native_allocate_real_3

   subroutine native_deallocate_real_1(this, ptr)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_real_1

   subroutine native_deallocate_real_2(this, ptr)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:,:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_real_2

   subroutine native_deallocate_real_3(this, ptr)
      class(NativeAllocator) :: this
      real(kind=C_DOUBLE), pointer, dimension(:,:,:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_real_3

! INTEGER data type
   subroutine native_allocate_int_1(this, ptr, dims)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1)) )
      return
   end subroutine native_allocate_int_1

   subroutine native_allocate_int_2(this, ptr, dims)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1), dims(2)) )
      return
   end subroutine native_allocate_int_2

   subroutine native_allocate_int_3(this, ptr, dims)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:,:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      allocate( ptr(dims(1), dims(2), dims(3)) )
      return
   end subroutine native_allocate_int_3

   subroutine native_deallocate_int_1(this, ptr)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_int_1

   subroutine native_deallocate_int_2(this, ptr)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:,:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_int_2

   subroutine native_deallocate_int_3(this, ptr)
      class(NativeAllocator) :: this
      integer(kind=C_INT), pointer, dimension(:,:,:), intent(inout) :: ptr

      deallocate( ptr )
      return
   end subroutine native_deallocate_int_3

!-----------------------------------------------------------------------------------------
! Page-locked memory allocator/deallocator, requires CUDA API.
!-----------------------------------------------------------------------------------------

! REAL data type
#if defined (USE_CUDA)
   subroutine pinned_allocate_real_1(this, ptr, dims)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      real(kind=c_double) :: dummy

      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )
      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return

   end subroutine pinned_allocate_real_1

   subroutine pinned_allocate_real_2(this, ptr, dims)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      real(kind=c_double) :: dummy
      
      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )
      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return

   end subroutine pinned_allocate_real_2

   subroutine pinned_allocate_real_3(this, ptr, dims)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:,:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      real(kind=c_double) :: dummy
      
      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )

      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return
   end subroutine pinned_allocate_real_3

   subroutine pinned_deallocate_real_1(this, ptr)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_real_1

   subroutine pinned_deallocate_real_2(this, ptr)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:,:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_real_2

   subroutine pinned_deallocate_real_3(this, ptr)
      class(PinnedAllocator) :: this
      real(kind=c_double), pointer, dimension(:,:,:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_real_3

! INTEGER data type
   subroutine pinned_allocate_int_1(this, ptr, dims)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      integer(kind=c_int) :: dummy

      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )
      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return

   end subroutine pinned_allocate_int_1

   subroutine pinned_allocate_int_2(this, ptr, dims)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      integer(kind=c_int) :: dummy
      
      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )
      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return

   end subroutine pinned_allocate_int_2

   subroutine pinned_allocate_int_3(this, ptr, dims)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:,:,:), intent(inout) :: ptr
      integer, dimension(:), intent(in) :: dims

      type(c_ptr) :: cptr
      integer :: err
      integer(kind=c_int) :: dummy
      
      err = cudaHostAlloc( cptr, sizeof(dummy) * product(dims), cudaHostAllocDefault )

      if (err == cudaSuccess) then
         call c_f_pointer( cptr, ptr, dims )
      endif
      return
   end subroutine pinned_allocate_int_3

   subroutine pinned_deallocate_int_1(this, ptr)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_int_1

   subroutine pinned_deallocate_int_2(this, ptr)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:,:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_int_2

   subroutine pinned_deallocate_int_3(this, ptr)
      class(PinnedAllocator) :: this
      integer(kind=c_int), pointer, dimension(:,:,:), intent(inout) :: ptr

      type(c_ptr) :: cptr
      integer :: err
      cptr = c_loc(ptr)

      err = cudaFreeHost(cptr)
      if (err == cudaSuccess) then
         nullify(ptr)
      endif
      return
   end subroutine pinned_deallocate_int_3
#endif

end module memory_allocators_mod

