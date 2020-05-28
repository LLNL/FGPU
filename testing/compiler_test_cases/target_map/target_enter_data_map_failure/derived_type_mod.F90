module derived_type_mod 

  type, public :: DerivedType 

     integer :: numg
     integer :: numa
     integer :: numc

     real, pointer :: weight(:)
     real, pointer :: psi(:,:,:)
     real, pointer :: phi(:,:)

  end type DerivedType 

! Test this out later
!!$omp declare target (derived_type_arr)
  type(DerivedType), pointer, public :: derived_type_arr(:)

contains

  subroutine construct_types(length)

    implicit none

    integer, intent(in) :: length
    integer :: i, a, c, g

    allocate(derived_type_arr(length))

    a = 64
    c = 16000
    g = 128

    do i=1, length 
      derived_type_arr(i)%numa = a
      derived_type_arr(i)%numc = c
      derived_type_arr(i)%numg = g

      allocate(derived_type_arr(i)%weight(a))
      allocate(derived_type_arr(i)%phi(g,c))
      allocate(derived_type_arr(i)%psi(g,c,a))

      derived_type_arr(i)%weight(:) = 1.0
      derived_type_arr(i)%phi(:,:) = 2.0
      derived_type_arr(i)%psi(:,:,:) = 3.0

    end do

  end subroutine construct_types


  subroutine alloc_on_device(self)
  
    implicit none

    type(DerivedType), intent(inout) :: self


!$omp target enter data map(alloc: self)
!$omp target enter data map(alloc: self%numg, self%numa, self%numc, self%weight, self%psi, self%phi) nowait

! File compiles if map is split into separate lines
!!$omp target enter data map(alloc: self%numg)
!!$omp target enter data map(alloc: self%numa)
!!$omp target enter data map(alloc: self%numc)
!!$omp target enter data map(alloc: self%weight)
!!$omp target enter data map(alloc: self%psi)
!!$omp target enter data map(alloc: self%phi)

  end subroutine alloc_on_device
 

  subroutine destroy_on_device(self)
  
    implicit none

    type(DerivedType), intent(inout) :: self

    !$omp target exit data map(delete: self%numg, self%numa, self%numc, self%weight, self%psi, self%phi) nowait

!!$omp target exit data map(delete: self%numg)
!!$omp target exit data map(delete: self%numa)
!!$omp target exit data map(delete: self%numc)

!!$omp target exit data map(delete: self%weight)
!!$omp target exit data map(delete: self%psi)
!!$omp target exit data map(delete: self%phi)
!$omp target exit data map(delete: self)


  end subroutine destroy_on_device

end module derived_type_mod
