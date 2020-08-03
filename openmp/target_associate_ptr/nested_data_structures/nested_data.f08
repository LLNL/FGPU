program fmain
   use example_types
   use openmp_tools
   use iso_c_binding

   implicit none
   integer :: i,n
   logical(C_BOOL) :: use_external_allocator

   use_external_allocator = .TRUE.

   call initialize()

   do i = 1, 1
      write(*,*) "------------- ITERATION ", i, " ----------------"
      typeQ_ptr%double =  i
      typeQ_ptr%double_array =  i

      do n=1,2
         typeQ_ptr%s_array(n)%double = i
         typeQ_ptr%s_array(n)%double_array = i
      enddo

      write(*,*) "\nOn host, before mapping to GPU."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      do n=1,2
      	write(*,*) "typeQ_ptr%s_array(", n, ")%double", typeQ_ptr%s_array(n)%double
	      write(*,*) "typeQ_ptr%s_array(", n, ")%double_array", typeQ_ptr%s_array(n)%double_array
		enddo

      print *, "--- mapping Q ---"     
      call map_to_typeQ_ptr(typeQ_ptr, use_external_allocator)
      print *, "--- mapping Q%double_array ---"     
      call map_to_double_1d(typeQ_ptr%double_array, use_external_allocator)

      print *, "--- mapping Q%s_array ---"     
      call map_to_typeS_1d(typeQ_ptr%s_array, use_external_allocator)
      print *, "--- mapping Q%s_array(1)%double_array ---"     
      call map_to_double_1d(typeQ_ptr%s_array(1)%double_array, use_external_allocator)
      print *, "--- mapping Q%s_array(2)%double_array ---"     
      call map_to_double_1d(typeQ_ptr%s_array(2)%double_array, use_external_allocator)
 
      !$omp target
      write(*,*) "\nOn device, after mapping to GPU"

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      write(*,*) "typeQ_ptr%s_array(1)%double", typeQ_ptr%s_array(1)%double
      write(*,*) "typeQ_ptr%s_array(1)%double_array", typeQ_ptr%s_array(1)%double_array
      write(*,*) "typeQ_ptr%s_array(2)%double", typeQ_ptr%s_array(2)%double
      write(*,*) "typeQ_ptr%s_array(2)%double_array", typeQ_ptr%s_array(2)%double_array

      typeQ_ptr%double =  0
      typeQ_ptr%double_array =  0

      do n=1,2
         typeQ_ptr%s_array(n)%double = 0
         typeQ_ptr%s_array(n)%double_array = 0
      enddo
      !$omp end target

      print *, "--- unmapping Q%s_array(2)%double_array ---"
      call map_exit_double_1d(typeQ_ptr%s_array(2)%double_array, use_external_allocator)
      print *, "--- unmapping Q%s_array(1)%double_array ---"     
      call map_exit_double_1d(typeQ_ptr%s_array(1)%double_array, use_external_allocator)
      print *, "--- unmapping Q%s_array ---"     
      call map_exit_typeS_1d(typeQ_ptr%s_array, use_external_allocator)

      print *, "--- unmapping Q%double_array ---"     
      call map_exit_double_1d(typeQ_ptr%double_array, use_external_allocator)
      print *, "--- unmapping Q ---"     
      call map_exit_typeQ_ptr(typeQ_ptr, use_external_allocator)
    
      write(*,*) "\nOn host, after mapping from GPU."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      do n=1,2
      	write(*,*) "typeQ_ptr%s_array(", n, ")%double", typeQ_ptr%s_array(n)%double
	      write(*,*) "typeQ_ptr%s_array(", n, ")%double_array", typeQ_ptr%s_array(n)%double_array
		enddo

   enddo

end program fmain
