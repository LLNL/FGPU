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

   len = 2
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

print *, "--- BEFORE MAP(TO:TYPE1_PTR) ---"
!$omp target enter data map(to:type1_ptr)
print *, "--- AFTER MAP(TO:TYPE1_PTR) ---"

print *, "--- BEFORE MAP(TO:TYPE1_PTR%SIMPLE_ARR) ---"
!$omp target enter data map(to:type1_ptr%simple_arr)
print *, "--- AFTER MAP(TO:TYPE1_PTR%SIMPLE_ARR) ---"

! Check with Tom on the OpenMP expected behavior when mapping an array of type pointers.
! In the IBM case, it appears to performs a deep copy ( will allocate and copy each type too ).

print *, "--- BEFORE MAP(TO:TYPE1_PTR%TYPE2_ARR) ---"
!$omp target enter data map(to:type1_ptr%type2_arr)
print *, "--- AFTER MAP(TO:TYPE1_PTR%TYPE2_ARR) ---"

   do n=1,len
      ! Appears to be superfluous for the IBM case, but verify behavior on other platforms
      ! when available. (should not have any ill effect, will just be redundant on IBM case)
      print *, "--- BEFORE MAP(TO:TYPE1_PTR%TYPE2_ARR(",n,") ---"
      !$omp target enter data map(to:type1_ptr%type2_arr(n))
      print *, "--- AFTER MAP(TO:TYPE1_PTR%TYPE2_ARR(",n,") ---"

      print *, "--- BEFORE MAP(TO:TYPE1_PTR%TYPE2_ARR(",n,")%SIMPLE_ARR ---"
      !$omp target enter data map(to:type1_ptr%type2_arr(n)%simple_arr)
      print *, "--- AFTER MAP(TO:TYPE1_PTR%TYPE2_ARR(",n,")%SIMPLE_ARR ---"
   end do

!$omp target
!   write(*,*) "\nOn device, before assignments.  Everything should be set to 1.0"
!   write(*,*) "\n---------------------------------------------------------------"
!   write(*,*) "type1%x", type1_ptr%x
!   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
!   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
!   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
!   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
!   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)

   type1_ptr%x = 2.0
   type1_ptr%simple_arr(1) = 2.0
   type1_ptr%type2_arr(1)%x = 2.0
   type1_ptr%type2_arr(1)%simple_arr(1) = 2.0

!   write(*,*) "\nIn device, after assignments."
!   write(*,*) "\n-----------------------------"
!   write(*,*) "\nFirst entry in type2_arr should be set to 2.0."
!   write(*,*) "type1%x", type1_ptr%x
!   write(*,*) "type1%simple_arr(1)", type1_ptr%simple_arr(1)
!   write(*,*) "type1%type2_arr(1)%x", type1_ptr%type2_arr(1)%x
!   write(*,*) "type1%type2_arr(1)%simple_arr(1)", type1_ptr%type2_arr(1)%simple_arr(1)
!   write(*,*) "\nSecond entry in type2_arr should be unchanged ( still 1.0 )."
!   write(*,*) "type1%type2_arr(2)%x", type1_ptr%type2_arr(2)%x
!   write(*,*) "type1%type2_arr(2)%simple_arr(1)", type1_ptr%type2_arr(2)%simple_arr(1)
!$omp end target 

! See note above about this being redundant ( at least for IBM behavior ).
   do n=1,len
      print *, "--- BEFORE MAP(FROM:TYPE1_PTR%TYPE2_ARR(",n,") ---"
      !$omp target exit data map(from:type1_ptr%type2_arr(n))
      print *, "--- AFTER MAP(FROM:TYPE1_PTR%TYPE2_ARR(",n,") ---"

      print *, "--- BEFORE MAP(FROM:TYPE1_PTR%TYPE2_ARR(",n,")%SIMPLE_ARR ---"
      !$omp target exit data map(from:type1_ptr%type2_arr(n)%simple_arr)
      print *, "--- AFTER MAP(FROM:TYPE1_PTR%TYPE2_ARR(",n,")%SIMPLE_ARR ---"
   end do

print *, "--- BEFORE MAP(FROM:TYPE1_PTR%TYPE2_ARR) ---"
!$omp target exit data map(from:type1_ptr%type2_arr)
print *, "--- AFTER MAP(FROM:TYPE1_PTR%TYPE2_ARR) ---"

print *, "--- BEFORE MAP(FROM:TYPE1_PTR%SIMPLE_ARR) ---"
!$omp target exit data map(from:type1_ptr%simple_arr)
print *, "--- AFTER MAP(FROM:TYPE1_PTR%SIMPLE_ARR) ---"

print *, "--- BEFORE MAP(FROM:TYPE1_PTR) ---"
!$omp target exit data map(from:type1_ptr)
print *, "--- AFTER MAP(FROM:TYPE1_PTR) ---"

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
