program fmain
   use example_types
   use openmp_tools
   use iso_c_binding

   implicit none
   integer :: i,n
   logical(C_BOOL) :: use_external_allocator
   logical(C_BOOL) :: dont_use_external_allocator

   use_external_allocator = .TRUE.
   dont_use_external_allocator = .FALSE.

   call initialize()

   do i = 1, 5
      write(*,*) "------------- ITERATION ", i, " ----------------"
      typeQ_ptr%double =  i
      typeQ_ptr%double_array =  i

      do n=1,2
         typeQ_ptr%s_array(n)%double = i
         typeQ_ptr%s_array(n)%double_array = i

         typeQ_ptr%g_array(n)%double = i
         typeQ_ptr%g_array(n)%double_array = i
      enddo

      write(*,*) "\nOn host, before mapping to GPU."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      write(*,*) "typeQ_ptr%s_array(1)%double", typeQ_ptr%s_array(1)%double
      write(*,*) "typeQ_ptr%s_array(1)%double_array", typeQ_ptr%s_array(1)%double_array
      write(*,*) "typeQ_ptr%s_array(2)%double", typeQ_ptr%s_array(2)%double
      write(*,*) "typeQ_ptr%s_array(2)%double_array", typeQ_ptr%s_array(2)%double_array

      write(*,*) "typeQ_ptr%g_array(1)%double", typeQ_ptr%g_array(1)%double
      write(*,*) "typeQ_ptr%g_array(1)%double_array", typeQ_ptr%g_array(1)%double_array
      write(*,*) "typeQ_ptr%g_array(2)%double", typeQ_ptr%g_array(2)%double
      write(*,*) "typeQ_ptr%g_array(2)%double_array", typeQ_ptr%g_array(2)%double_array

      ! Map over 'Q' derived type
      
      call map_to_typeQ(typeQ_ptr, use_external_allocator)
      !call map_to_typeQ(typeQ_ptr, dont_use_external_allocator) ! <- works ok

      !---  The rest of these map to's appear to be working fine. -----
      call map_to_double_1d(typeQ_ptr%double_array, use_external_allocator)

      ! Map over array of 'S' derived types in Q.
      call map_to_typeS_1d(typeQ_ptr%s_array, use_external_allocator)
      call map_to_double_1d(typeQ_ptr%s_array(1)%double_array, use_external_allocator)
      call map_to_double_1d(typeQ_ptr%s_array(2)%double_array, use_external_allocator)
 
      ! Map over array of 'G' derived types in Q
      call map_to_typeG_1d(typeQ_ptr%g_array, use_external_allocator)
      call map_to_double_1d(typeQ_ptr%g_array(1)%double_array, use_external_allocator)
      call map_to_double_1d(typeQ_ptr%g_array(2)%double_array, use_external_allocator)

      !$omp target
      write(*,*) "\nOn device, after mapping to GPU"

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      write(*,*) "typeQ_ptr%s_array(1)%double", typeQ_ptr%s_array(1)%double
      write(*,*) "typeQ_ptr%s_array(1)%double_array", typeQ_ptr%s_array(1)%double_array
      write(*,*) "typeQ_ptr%s_array(2)%double", typeQ_ptr%s_array(2)%double
      write(*,*) "typeQ_ptr%s_array(2)%double_array", typeQ_ptr%s_array(2)%double_array

      write(*,*) "typeQ_ptr%g_array(1)%double", typeQ_ptr%g_array(1)%double
      write(*,*) "typeQ_ptr%g_array(1)%double_array", typeQ_ptr%g_array(1)%double_array
      write(*,*) "typeQ_ptr%g_array(2)%double", typeQ_ptr%g_array(2)%double
      write(*,*) "typeQ_ptr%g_array(2)%double_array", typeQ_ptr%g_array(2)%double_array

      typeQ_ptr%double =  0
      typeQ_ptr%double_array =  0

      do n=1,2
         typeQ_ptr%s_array(n)%double = 0
         typeQ_ptr%s_array(n)%double_array = 0

         typeQ_ptr%g_array(n)%double = 0
         typeQ_ptr%g_array(n)%double_array = 0
      enddo
      !$omp end target

!---  Most of the map exit's appear to work fine. ------

      ! Map back array of 'G' derived types in Q
      call map_exit_double_1d(typeQ_ptr%g_array(2)%double_array, use_external_allocator)
      call map_exit_double_1d(typeQ_ptr%g_array(1)%double_array, use_external_allocator)
      call map_exit_typeG_1d(typeQ_ptr%g_array, use_external_allocator)

      ! Map back array of 'S' derived types in Q
      call map_exit_double_1d(typeQ_ptr%s_array(2)%double_array, use_external_allocator)
      call map_exit_double_1d(typeQ_ptr%s_array(1)%double_array, use_external_allocator)
      call map_exit_typeS_1d(typeQ_ptr%s_array, use_external_allocator)

      
      ! Map back 'Q' derived type
      call map_exit_double_1d(typeQ_ptr%double_array, use_external_allocator)

!--- This last map exit below fails with a
!LOMP: warning in "memcopy frin device"
!1587-175 The underlying GPU runtime reported the following error "invalid argument". 
!LOMP: error in "memcopy frin device"
!1587-163 Error encountered while attempting to execute on the target device 0.  The program will stop.

      call map_exit_typeQ(typeQ_ptr, use_external_allocator)
      !call map_exit_typeQ(typeQ_ptr, dont_use_external_allocator) ! <- works ok
    
      write(*,*) "\nOn host, after mapping from GPU."

      write(*,*) "typeQ_ptr%double", typeQ_ptr%double
      write(*,*) "typeQ_ptr%double_array", typeQ_ptr%double_array

      write(*,*) "typeQ_ptr%s_array(1)%double", typeQ_ptr%s_array(1)%double
      write(*,*) "typeQ_ptr%s_array(1)%double_array", typeQ_ptr%s_array(1)%double_array
      write(*,*) "typeQ_ptr%s_array(2)%double", typeQ_ptr%s_array(2)%double
      write(*,*) "typeQ_ptr%s_array(2)%double_array", typeQ_ptr%s_array(2)%double_array

      write(*,*) "typeQ_ptr%g_array(1)%double", typeQ_ptr%g_array(1)%double
      write(*,*) "typeQ_ptr%g_array(1)%double_array", typeQ_ptr%g_array(1)%double_array
      write(*,*) "typeQ_ptr%g_array(2)%double", typeQ_ptr%g_array(2)%double
      write(*,*) "typeQ_ptr%g_array(2)%double_array", typeQ_ptr%g_array(2)%double_array

   enddo

end program fmain
