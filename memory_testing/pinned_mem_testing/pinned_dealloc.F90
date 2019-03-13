program main

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

   print *, "Sum: ", SUM(phi(:,:))

   deallocate(weight)
   deallocate(phi)
   deallocate(psi)

   return

end program main
