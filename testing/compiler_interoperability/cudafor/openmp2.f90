         program simple
         integer onGPU

         onGPU = 0;
!$omp target map(from:onGPU)
         if (omp_is_initial_device() .eq. 0) then
              onGPU = 1
         else
              onGPU = 2
         end if 
!$omp end target
         if (onGPU .eq. 1) then
            print *, 'Fortran is running on the GPU', onGPU
         else
            print *, 'Fortran is NOT running on the GPU', onGPU
         end if 


         print *, 'Exiting program for fortran'
         end
