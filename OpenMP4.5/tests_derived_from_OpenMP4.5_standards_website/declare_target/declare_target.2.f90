program my_main
   call foo()
end program

subroutine foo()
	use cudafor

   integer, pointer, managed :: N(:)
   !$omp declare target(fib)

	allocate(N(100))


   !$omp target
   call fib(N)
   !$omp end target
end subroutine

subroutine fib(N)
	integer, pointer, intent(in) :: N(:)
	!$omp declare target

	  write(*,*) "hello from fib"

end subroutine
