module omp45_kernel_mod
contains

   subroutine omp45_kernel_test(id)
   use derived_type_mod

   implicit none


   integer, intent(in) :: id
   type(DerivedType), pointer :: derived_type

   integer    :: a, c, g, numa, numc, numg, length

   derived_type => derived_type_arr(id)

   numa = derived_type%numa
   numg = derived_type%numg
   numc = derived_type%numc

!!$omp target map(to:a,c,g,numa,numc,numg,derived_type%weight,derived_type%psi) map(tofrom:derived_type%phi)

!$omp target map(to:c,g,a, numa, numc, numg)
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

   print *, "Ran looptest openmp4.5 test, SUM: ", SUM(derived_type%phi(:,:))
   return


   end subroutine omp45_kernel_test

end module omp45_kernel_mod
