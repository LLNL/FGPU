program main

   use cudafor
   use iso_c_binding

   implicit none

!  Arguments

   integer    :: a, c, g, err
   integer    :: numa, numc, numg

   real, pointer :: weight(:)
   real, pointer :: phi(:,:)
   real, pointer :: psi(:,:,:)

   type(C_PTR) :: cptr
  
   numa = 64
   numc = 16000
   numg = 128

   err = cudaHostAlloc( cptr, sizeof(0.0) * numa, cudaHostAllocDefault )
   call c_f_pointer( cptr, weight, [numa] )

   err = cudaHostAlloc( cptr, sizeof(0.0) * numg * numc, cudaHostAllocDefault )
   call c_f_pointer( cptr, phi, [numg,numc] )

   err = cudaHostAlloc( cptr, sizeof(0.0) * numg * numc * numa, cudaHostAllocDefault )
   call c_f_pointer( cptr, psi, [numg,numc,numa] )

!   allocate(weight(numa))
!   allocate(phi(numg,numc))
!   allocate(psi(numg,numc,numa))

   weight(:) = 1.0
   phi(:,:) = 2.0
   psi(:,:,:) = 3.0

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

   print *, "Sum: ", SUM(phi(:,:))
   return

end program main
