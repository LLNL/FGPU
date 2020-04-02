module derived_type_mod

type, public :: DerivedType

integer :: anInt
real :: aReal

real, pointer, contiguous :: arr1(:)
real, pointer, contiguous :: arr2(:,:)
real, pointer, contiguous :: arr3(:,:,:)

end type DerivedType

!$omp declare target(dt_ptr)
type(DerivedType), pointer, public :: dt_ptr

contains

subroutine dt_ctor(self)
implicit none

type(DerivedType), intent(inout) :: self

self% anInt = 100
self% aReal = 10.0

allocate(self%arr1(1000))
allocate(self%arr2(100,150))
allocate(self%arr3(5, 150, 10))

self%arr1(1) = 1.0
self%arr2(1,1) = 1.0
self%arr3(1,1,1) = 1.0

!$omp target enter data map(alloc:dt_ptr)
!$omp target enter data map(alloc:dt_ptr%arr1, dt_ptr%arr2, dt_ptr%arr3)

end subroutine dt_ctor

subroutine dt_dtor(self)
implicit none

type(DerivedType), intent(inout) :: self

!$omp target exit data map(delete:self%arr1, self%arr2, self%arr3)
!$omp target exit data map(delete:self)

deallocate(self%arr1)
deallocate(self%arr2)
deallocate(self%arr3)

end subroutine dt_dtor
end module derived_type_mod
