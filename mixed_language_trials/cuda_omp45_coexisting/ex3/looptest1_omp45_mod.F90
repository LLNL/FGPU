module looptest1_omp45_mod
contains

   subroutine looptest1_omp45()

   use cudafor
   implicit none

!  Arguments

   integer    :: a, c, g
   integer    :: numa, numc, numg
   real, allocatable, pinned :: weight(:)
   real, allocatable, pinned :: phi(:,:)
   real, allocatable, pinned :: psi(:,:,:)

   numa = 64
   numc = 16000
   numg = 128

   allocate(weight(numa))
   allocate(phi(numg,numc))
   allocate(psi(numg,numc,numa))

   weight(:) = 1.0
   phi(:,:) = 2.0
   psi(:,:,:) = 3.0

!  Same loop hierarchy as CUDA version.  Should use coalesced memory in inner
!  loop, as all threads are accessing same angle's worth chunk of memory.
!  Teams (blocks) and threads explicitly set, same as CUDA.

!$omp target map(to:a,c,g,numa,numc,numg,weight,psi) map(tofrom:phi)
!$omp teams num_teams(numc) thread_limit(numg)
!$omp distribute parallel do collapse(2)
   do c=1,numc
    do g=1,numg
       do a=1,numa
         phi(g, c) = phi(g,c) + (weight(a) * psi(g,c,a))
       end do
     end do
   end do
!$omp end distribute parallel do
!$omp end teams
!$omp end target

   print *, "Ran looptest openmp4.5 test."
   print *, "Sum: ", SUM(phi(:,:))
   return

   end subroutine looptest1_omp45

end module looptest1_omp45_mod
