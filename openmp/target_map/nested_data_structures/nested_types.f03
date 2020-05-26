module example

  type, public :: type2
    real(8)              :: x
    real(8), pointer     :: simple_arr(:)
  end type type2

  type, public :: type1
    real(8)              :: x
    real(8), pointer     :: simple_arr(:)
    type(type2), pointer :: type2_arr(:)
  end type type1

  type(type1), pointer, public :: type1_ptr

  contains

    subroutine initialize(len)
      implicit none
      integer, intent(in) :: len
      integer :: n

      type1_ptr%x = 1.0

      allocate(type1_ptr%simple_arr(10))
      type1_ptr%simple_arr(:) = 1.0

      allocate(type1_ptr%type2_arr(len))

      do n=1,len
        type1_ptr%type2_arr(n)%x = 2.0
        allocate(type1_ptr%type2_arr(n)%simple_arr(10))
        type1_ptr%type2_arr(n)%simple_arr(:) = 2.0
      end do

    end subroutine initialize

end module example


program fmain
   use example

   implicit none
   integer :: n, len

   len = 5
   allocate(type1_ptr)
   call initialize(len)

   write(*,*) "\nOn host, before mapping data to device."

   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)

!$omp target enter data map(to:type1_ptr)
!$omp target enter data map(to:type1_ptr%simple_arr)
!$omp target enter data map(to:type1_ptr%type2_arr)

   do n=1,len
      !$omp target enter data map(to:type1_ptr%type2_arr(n))
      !$omp target enter data map(to:type1_ptr%type2_arr(n)%simple_arr)
   end do

!$omp target
   write(*,*) "\nOn device, before assignments."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)

   type1_ptr%x = 10.0
   type1_ptr%simple_arr(1) = 10.0
   type1_ptr%type2_arr(1)%x = 10.0
   type1_ptr%type2_arr(1)%simple_arr(1) = 10.0

   write(*,*) "\nIn device, after assignments."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "(next two should be unchanged)"
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)
!$omp end target 

   do n=1,len
      !$omp target exit data map(from:type1_ptr%type2_arr(n)%simple_arr)
      !$omp target exit data map(from:type1_ptr%type2_arr(n))
   end do

!$omp target exit data map(from:type1_ptr%type2_arr)
!$omp target exit data map(from:type1_ptr%simple_arr)
!$omp target exit data map(from:type1_ptr)

   write(*,*) "\nOn host, after map back."
   write(*,*) "These should be 10."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "Next two should be unchanged"
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)

end program fmain
