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
        type1_ptr%type2_arr(n)%x = 1.0
        allocate(type1_ptr%type2_arr(n)%simple_arr(10))
        type1_ptr%type2_arr(n)%simple_arr(:) = 1.0
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

   write(*,*) "\nOn host, before mapping data to device.  Everything is set to 1.0."
   write(*,*) "\n------------------------------------------------------------------"
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)

!$omp target enter data map(to:type1_ptr)
!$omp target enter data map(to:type1_ptr%simple_arr)

! Check with Tom on the OpenMP expected behavior when mapping an array of type pointers.
! In the IBM case, it appears to performs a deep copy ( will allocate and copy each type too ).
!$omp target enter data map(to:type1_ptr%type2_arr)

! Appears to be superfluous for the IBM case, but verify behavior on other platforms
! when available. (should not have any ill effect, will just be redundant on IBM case)
   do n=1,len
      !$omp target enter data map(to:type1_ptr%type2_arr(n)%simple_arr)
   end do

!$omp target
   write(*,*) "\nOn device, before assignments.  Everything should be set to 1.0"
   write(*,*) "\n---------------------------------------------------------------"
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)

   type1_ptr%x = 2.0
   type1_ptr%simple_arr(1) = 2.0
   type1_ptr%type2_arr(1)%x = 2.0
   type1_ptr%type2_arr(1)%simple_arr(1) = 2.0

   write(*,*) "\nIn device, after assignments."
   write(*,*) "\n-----------------------------"
   write(*,*) "\nFirst entry in type2_arr should be set to 2.0."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "\nSecond entry in type2_arr should be unchanged ( still 1.0 )."
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)
!$omp end target 

! See note at line 63 about this being redundant ( at least for IBM behavior ).
   do n=1,len
      !$omp target exit data map(from:type1_ptr%type2_arr(n)%simple_arr)
   end do

!$omp target exit data map(from:type1_ptr%type2_arr)
!$omp target exit data map(from:type1_ptr%simple_arr)
!$omp target exit data map(from:type1_ptr)

   write(*,*) "\nOn host, after map back."
   write(*,*) "\n------------------------"
   write(*,*) "Type 1 vars should be 2.0."
   write(*,*) "type1%x", type1_ptr%x
   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
   write(*,*) "\nFirst entry in type2_arr should be 2.0."
   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
   write(*,*) "\nSecond entry in type2_arr should be unchanged ( still 1.0 )."
   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)

end program fmain
