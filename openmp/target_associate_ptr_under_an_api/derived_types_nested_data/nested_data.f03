module example
   use iso_c_binding

  type, public :: typeS
    real(C_DOUBLE)                            :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
  end type typeS

  type, public :: typeG
    real(C_DOUBLE)                        :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
  end type typeG

  type, public :: typeQ
    real(C_DOUBLE)                        :: double
    real(C_DOUBLE), pointer, dimension(:) :: double_array
    type(typeS), pointer, dimension(:)    :: s_array
    type(typeG), pointer, dimension(:)    :: g_array
  end type typeQ

  type(typeQ), pointer, public :: typeQ_ptr

  contains

    subroutine initialize()
      implicit none
      integer :: n

      allocate(typeQ_ptr)
      allocate(typeQ_ptr%double_array(10))

      allocate(typeQ_ptr%s_array(2))
      allocate(typeQ_ptr%g_array(2))

      do n=1,2
         allocate(typeQ_ptr%s_array(n)%double_array(5))
         allocate(typeQ_ptr%g_array(n)%double_array(5))
      enddo

    end subroutine initialize

end module example


program fmain
   use example
!   use openmp_tools
   use iso_c_binding

   implicit none
   integer :: i,j,n
   logical(C_BOOL) :: use_external_allocator

   use_external_allocator = .TRUE.

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


      !$omp target enter data map(to:typeQ_ptr)
!      !$omp target enter data map(to:typeQ_ptr%double_array)
      !$omp target
      typeQ_ptr%double =  0
      typeQ_ptr%double_array =  0
      do n=1,2
         typeQ_ptr%s_array(n)%double = 0
         typeQ_ptr%s_array(n)%double_array = 0

         typeQ_ptr%g_array(n)%double = 0
         typeQ_ptr%g_array(n)%double_array = 0
      enddo
      !$omp end target

!      !$omp target exit data map(from:typeQ_ptr%double_array)
      !$omp target exit data map(from:typeQ_ptr)
    

      ! Map back array of 'S' derived types in Q
      
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
