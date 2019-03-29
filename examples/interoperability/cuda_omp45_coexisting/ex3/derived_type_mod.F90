module derived_type_mod 

  type, public :: DerivedType 

     integer :: numg
     integer :: numa
     integer :: numc

     real, allocatable :: weight(:)
     real, allocatable :: psi(:,:,:)
     real, allocatable :: phi(:,:)

  end type DerivedType 

end module derived_type_mod
