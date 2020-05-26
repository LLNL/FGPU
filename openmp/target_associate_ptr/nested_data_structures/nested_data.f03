module example
   use iso_c_binding

  type, public :: type2
    real(C_DOUBLE)              :: x
    real(C_DOUBLE), pointer     :: simple_arr(:,:,:)
  end type type2

  type, public :: type1
    real(C_DOUBLE)              :: x
    real(C_DOUBLE), pointer     :: simple_arr(:,:,:)
    type(type2), pointer :: type2_arr(:)
  end type type1

  type(type1), pointer, public :: type1_ptr

  contains

    subroutine initialize(len)
      implicit none
      integer, intent(in) :: len
      integer :: n

      type1_ptr%x = 1.0

      allocate(type1_ptr%simple_arr(2,2,2))
      type1_ptr%simple_arr = 1.0

      allocate(type1_ptr%type2_arr(len))

      do n=1,len
        type1_ptr%type2_arr(n)%x = 2.0
        allocate(type1_ptr%type2_arr(n)%simple_arr(2,2,2))
        type1_ptr%type2_arr(n)%simple_arr = 2.0
      end do

    end subroutine initialize

end module example


program fmain
   use example
   use openmp_tools
	use iso_c_binding

   implicit none
   integer :: n, len
	logical(C_BOOL) :: use_external_allocator

	use_external_allocator = .FALSE.
   len = 5
   allocate(type1_ptr)
   call initialize(len)

   write(*,*) "\nOn host, before map to."

   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1,1,1)", type1_ptr%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1,1,1)", type1_ptr%type2_arr(1)%simple_arr(1,1,1)

!$omp target enter data map(to:type1_ptr)

!   call map_alloc(type1_ptr%simple_arr, use_external_allocator)
!   !$omp target update to(type1_ptr%simple_arr)
!$omp target enter data map(to:type1_ptr%simple_arr)

!$omp target enter data map(to:type1_ptr%type2_arr)

   do n=1,len
      !$omp target enter data map(to:type1_ptr%type2_arr(n))

!      call map_alloc(type1_ptr%type2_arr(n)%simple_arr, use_external_allocator)
!      !$omp target update to(type1_ptr%type2_arr(n)%simple_arr)

		!$omp target enter data map(to:type1_ptr%type2_arr(n)%simple_arr)
   end do

!$omp target
   write(*,*) "\nOn device, before assignments."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1,1,1)", type1_ptr%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1,1,1)", type1_ptr%type2_arr(1)%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1,1,1)", type1_ptr%type2_arr(2)%simple_arr(1,1,1)

   type1_ptr%x = 10.0
   type1_ptr%simple_arr(1,1,1) = 10.0
   type1_ptr%type2_arr(1)%x = 10.0
   type1_ptr%type2_arr(1)%simple_arr(1,1,1) = 10.0

   write(*,*) "\nIn device, after assignments."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1,1,1)", type1_ptr%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1,1,1)", type1_ptr%type2_arr(1)%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1,1,1)", type1_ptr%type2_arr(2)%simple_arr(1,1,1)
!$omp end target 

   do n=1,len
!      !$omp target update from(type1_ptr%type2_arr(n)%simple_arr)
!      call map_delete(type1_ptr%type2_arr(n)%simple_arr, use_external_allocator)

		!$omp target exit data map(from:type1_ptr%type2_arr(n)%simple_arr)
      !$omp target exit data map(from:type1_ptr%type2_arr(n))
   end do

!$omp target exit data map(from:type1_ptr%type2_arr)

!   !$omp target update from(type1_ptr%simple_arr)
!   call map_delete(type1_ptr%simple_arr, use_external_allocator)

!$omp target exit data map(from:type1_ptr%simple_arr)
!$omp target exit data map(from:type1_ptr)

   write(*,*) "\nOn host, after map back."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1,1,1)", type1_ptr%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1,1,1)", type1_ptr%type2_arr(1)%simple_arr(1,1,1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1,1,1)", type1_ptr%type2_arr(2)%simple_arr(1,1,1)

end program fmain
