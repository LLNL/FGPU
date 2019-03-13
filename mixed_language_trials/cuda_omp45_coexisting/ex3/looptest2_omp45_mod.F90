module looptest2_omp45_mod
contains

   subroutine looptest2_omp45()

   use cudafor
   use derived_type_mod
   implicit none

   integer    :: a, c, g, numa, numc, numg

!   type:: DerivedType
!
!      integer :: numa, numc, numg
!      real, allocatable :: weight(:), phi(:,:), psi(:,:,:)
!
!   end type DerivedType

   type(DerivedType) :: derived_type

   derived_type%numa = 64
   derived_type%numc = 16000
   derived_type%numg = 128

   allocate(derived_type%weight(derived_type%numa))
   allocate(derived_type%phi(derived_type%numg,derived_type%numc))
   allocate(derived_type%psi(derived_type%numg,derived_type%numc,derived_type%numa))

   derived_type%weight(:) = 1.0
   derived_type%phi(:,:) = 2.0
   derived_type%psi(:,:,:) = 3.0

   numa = derived_type%numa
   numg = derived_type%numg
   numc = derived_type%numc

!$omp target map(to:a,c,g,numa,numc,numg,derived_type%weight,derived_type%psi) map(tofrom:derived_type%phi)
!$omp teams num_teams(numc) thread_limit(numg)
!$omp distribute parallel do collapse(2)
   do c=1,numc
    do g=1,numg
       do a=1,numa
         derived_type%phi(g, c) = derived_type%phi(g,c) + (derived_type%weight(a) * derived_type%psi(g,c,a))
       end do
     end do
   end do
!$omp end distribute parallel do
!$omp end teams
!$omp end target

   print *, "Ran looptest openmp4.5 test."
   print *, "Sum: ", SUM(derived_type%phi(:,:))
   return

   end subroutine looptest2_omp45

end module looptest2_omp45_mod
