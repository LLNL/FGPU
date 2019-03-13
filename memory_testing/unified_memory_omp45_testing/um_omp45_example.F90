program my_main
   call foo()
end program

subroutine foo()
   use cudafor

   integer, pointer, managed :: N(:)
   !$omp declare target(bar)

   allocate(N(100))

   !$omp target map(N)
   call bar(N)
   !$omp end target

   write(*,*) "Host: N(1)", N(1)
end subroutine

subroutine bar(N)
   integer, pointer :: N(:)
   !$omp declare target

   N(1) = 1.0
   write(*,*) "Device: N(1)", N(1)

end subroutine
