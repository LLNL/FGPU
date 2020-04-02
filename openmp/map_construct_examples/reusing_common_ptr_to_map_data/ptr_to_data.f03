module data
   double precision, target, dimension(10,10) :: a, b
   double precision, pointer, contiguous :: data_ptr(:,:)
end module data

program map_testing
   use data
   implicit none

	a = 1.0
   b = 2.0

   data_ptr => a
   print *, "----------- Mapping a via data_ptr----------"
   !$omp target enter data map(to:data_ptr)

   data_ptr => b
   print *, "----------- Mapping b via data_ptr----------"
   !$omp target enter data map(to:data_ptr)

	!$omp target
	write (*,*) "After enter data map, on device: a(1,1): ", a(1,1)
	write (*,*) "After enter data map, on device: b(1,1): ", b(1,1)

	a(1,1) = 10.0
	b(1,1) = 20.0

	write (*,*) "After device update: a(1,1): ", a(1,1)
	write (*,*) "After device update: b(1,1): ", b(1,1)
	!$omp end target

	data_ptr=>a
	!$omp target exit data map(from:data_ptr)
	data_ptr=>b
	!$omp target exit data map(from:data_ptr)

	write (*,*) "After exit data map, on host: a(1,1): ", a(1,1)
	write (*,*) "After exit data map, on host: b(1,1): ", b(1,1)



end program map_testing
