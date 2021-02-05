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
         typeQ_ptr%s_array(n)%double_array_2d = i
         typeQ_ptr%s_array(n)%double_array_3d = i
      enddo

      write(*,*) "\nOn host, before mapping to GPU."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      do n=1,2
         write(*,*) "typeQ_ptr%s_array(", n, ")%double", typeQ_ptr%s_array(n)%double
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array", typeQ_ptr%s_array(n)%double_array
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array_2d", typeQ_ptr%s_array(n)%double_array_2d
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array_3d", typeQ_ptr%s_array(n)%double_array_3d
      enddo

      print *, "--- mapping Q ---"
      if (use_external_allocator) then
         call target_alloc_and_associate_typeQ_ptr(typeQ_ptr)
      endif
      !$omp target enter data map(always, to:typeQ_ptr)

      print *, "--- mapping Q%double_array ---"     
      if (use_external_allocator) then
         call target_alloc_and_associate_double_1d(typeQ_ptr%double_array)
      endif
      !$omp target enter data map(always, to:typeQ_ptr%double_array)

      print *, "--- mapping Q%s_array ---"
      if (use_external_allocator) then
         call target_alloc_and_associate_typeS_1d(typeQ_ptr%s_array)
      endif
      !$omp target enter data map(always, to:typeQ_ptr%s_array)

      print *, "--- mapping Q%s_array(1) double_arrays ---"     
      if (use_external_allocator) then
         call target_alloc_and_associate_double_1d(typeQ_ptr%s_array(1)%double_array)
         call target_alloc_and_associate_double_2d(typeQ_ptr%s_array(1)%double_array_2d)
         call target_alloc_and_associate_double_3d(typeQ_ptr%s_array(1)%double_array_3d)
      endif
         !$omp target enter data map(always,to:typeQ_ptr%s_array(1)%double_array)
         !$omp target enter data map(always,to:typeQ_ptr%s_array(1)%double_array_2d)
         !$omp target enter data map(always,to:typeQ_ptr%s_array(1)%double_array_3d)

      print *, "--- mapping Q%s_array(2) double_arrays  ---"     
      if (use_external_allocator) then
         call target_alloc_and_associate_double_1d(typeQ_ptr%s_array(2)%double_array)
         call target_alloc_and_associate_double_2d(typeQ_ptr%s_array(2)%double_array_2d)
         call target_alloc_and_associate_double_3d(typeQ_ptr%s_array(2)%double_array_3d)
      endif
      !$omp target enter data map(always,to:typeQ_ptr%s_array(2)%double_array)
      !$omp target enter data map(always,to:typeQ_ptr%s_array(2)%double_array_2d)
      !$omp target enter data map(always,to:typeQ_ptr%s_array(2)%double_array_3d)
 
      !$omp target
      write(*,*) "\nOn device, after mapping to GPU"

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      write(*,*) "typeQ_ptr%s_array(1)%double", typeQ_ptr%s_array(1)%double
      write(*,*) "typeQ_ptr%s_array(1)%double_array", typeQ_ptr%s_array(1)%double_array
      write(*,*) "typeQ_ptr%s_array(1)%double_array_2d", typeQ_ptr%s_array(1)%double_array_2d
      write(*,*) "typeQ_ptr%s_array(1)%double_array_3d", typeQ_ptr%s_array(1)%double_array_3d
      write(*,*) "typeQ_ptr%s_array(2)%double", typeQ_ptr%s_array(2)%double
      write(*,*) "typeQ_ptr%s_array(2)%double_array", typeQ_ptr%s_array(2)%double_array
      write(*,*) "typeQ_ptr%s_array(2)%double_array_2d", typeQ_ptr%s_array(2)%double_array_2d
      write(*,*) "typeQ_ptr%s_array(2)%double_array_3d", typeQ_ptr%s_array(2)%double_array_3d

      typeQ_ptr%double =  0
      typeQ_ptr%double_array =  0

      do n=1,2
         typeQ_ptr%s_array(n)%double = 0
         typeQ_ptr%s_array(n)%double_array = 0
         typeQ_ptr%s_array(n)%double_array_2d = 0
         typeQ_ptr%s_array(n)%double_array_3d = 0
      enddo
      !$omp end target

      print *, "--- unmapping Q%s_array(2) double_arrays ---"
      !$omp target exit data map(always,from:typeQ_ptr%s_array(2)%double_array)
      !$omp target exit data map(always,from:typeQ_ptr%s_array(2)%double_array_2d)
      !$omp target exit data map(always,from:typeQ_ptr%s_array(2)%double_array_3d)
      if (use_external_allocator) then
         call target_free_and_disassociate_double_1d(typeQ_ptr%s_array(2)%double_array)
         call target_free_and_disassociate_double_2d(typeQ_ptr%s_array(2)%double_array_2d)
         call target_free_and_disassociate_double_3d(typeQ_ptr%s_array(2)%double_array_3d)
      endif

      print *, "--- unmapping Q%s_array(1) double_arrays ---"     
      !$omp target exit data map(always,from:typeQ_ptr%s_array(1)%double_array)
      !$omp target exit data map(always,from:typeQ_ptr%s_array(1)%double_array_2d)
      !$omp target exit data map(always,from:typeQ_ptr%s_array(1)%double_array_3d)
      if (use_external_allocator) then
         call target_free_and_disassociate_double_1d(typeQ_ptr%s_array(1)%double_array)
         call target_free_and_disassociate_double_2d(typeQ_ptr%s_array(1)%double_array_2d)
         call target_free_and_disassociate_double_3d(typeQ_ptr%s_array(1)%double_array_3d)
      endif

      print *, "--- unmapping Q%s_array ---"     
      !$omp target exit data map(always,from:typeQ_ptr%s_array)
      if (use_external_allocator) then
         call target_free_and_disassociate_typeS_1d(typeQ_ptr%s_array)
      endif

      print *, "--- unmapping Q%double_array ---"     
      !$omp target exit data map(always,from:typeQ_ptr%double_array)
      if (use_external_allocator) then
         call target_free_and_disassociate_double_1d(typeQ_ptr%double_array)
      endif

      print *, "--- unmapping Q ---"     
      !$omp target exit data map(always,from:typeQ_ptr)
      if (use_external_allocator) then
         call target_free_and_disassociate_typeQ_ptr(typeQ_ptr)
      endif
    
      write(*,*) "\nOn host, after mapping from GPU."
      write(*,*) "\nAll values below should be 0."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      do n=1,2
         write(*,*) "typeQ_ptr%s_array(", n, ")%double", typeQ_ptr%s_array(n)%double
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array", typeQ_ptr%s_array(n)%double_array
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array_2d", typeQ_ptr%s_array(n)%double_array_2d
         write(*,*) "typeQ_ptr%s_array(", n, ")%double_array_3d", typeQ_ptr%s_array(n)%double_array_3d
      enddo
   enddo

end program fmain
