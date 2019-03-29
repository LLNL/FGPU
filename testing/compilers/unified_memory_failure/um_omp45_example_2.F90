module numbers

   type, public :: some_numbers
      integer :: length
   end type some_numbers

   type(some_numbers), pointer, public :: somenum

end module numbers

program my_main
   use numbers

   allocate(somenum)
   somenum%length = 17607

   call foo()
   
end program


subroutine foo()
   use cudafor
   use numbers

   integer, managed :: N0(128)
   integer, managed :: N1(128)
   real, managed :: N2(128)
   real, managed :: N3(somenum%length)
   real, managed :: N4(18849)
   integer :: i, len0, len2

   len0 = 128
   len2 = 849
   N3(1) = 0

   write(*,*) "Host: N3(1)", N3(1)

   !$omp target
   N3(1) = 1.0
   write(*,*) "Device: N3(1)", N3(1)
   !$omp end target

   write(*,*) "Host: N3(1)", N3(1)

   do i=1, 1000
   !$omp target
   call bar(len0, somenum%length, len2,N0,N1,N2,N3,N4)
   !$omp end target
   end do

   write(*,*) "Host: N3(1)", N3(1)

end subroutine


subroutine bar(len0, len1, len2, N0, N1, N2, N3, N4)
   integer, intent(in), managed :: N0(len0)
   integer, intent(in), managed :: N1(len0)
   real, intent(inout), managed :: N2(len0)
   real, intent(inout), managed :: N3(len1)
   real, intent(inout), managed :: N4(len2)
   !$omp declare target

   N3(1) = 1.0

   return

end subroutine
