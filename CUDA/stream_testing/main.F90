   program fmain

   use derived_type_mod
   use cuda_test_mod
   use cudafor

   integer OMP_GET_MAX_THREADS

   integer :: num_max_threads, i, numg, numa, numc

   type(DerivedType), allocatable :: type_arr(:)

   num_max_threads = OMP_GET_MAX_THREADS()
   print *, "fmain: number of max threads: ", num_max_threads

   numa = 64
   numc = 16000
   numg = 128

   print *,"Max number of threads: ", num_max_threads

   allocate(type_arr(num_max_threads))

!$omp parallel do private(i) schedule(dynamic)
   do i=1,num_max_threads
      istat = cudaStreamCreate(type_arr(i)%stream)

      ! Set stream so the device allocations are async in this loop.
      istats = cudaforSetDefaultStream(type_arr(i)%stream)

      type_arr(i)%numa = numa
      type_arr(i)%numc = numc
      type_arr(i)%numg = numg

      allocate(type_arr(i)%weight(numa))
      allocate(type_arr(i)%phi(numg,numc))
      allocate(type_arr(i)%psi(numg,numc,numa))

      allocate(type_arr(i)%weight_d(numa))
      allocate(type_arr(i)%phi_d(numg,numc))
      allocate(type_arr(i)%psi_d(numg,numc,numa))

      istat = cudaforSetDefaultStream(type_arr(i)%weight_d, type_arr(i)%stream)
      istat = cudaforSetDefaultStream(type_arr(i)%phi_d, type_arr(i)%stream)
      istat = cudaforSetDefaultStream(type_arr(i)%psi_d, type_arr(i)%stream)

   enddo
!$omp end parallel do

!$omp parallel do private(n)
   do i=1,num_max_threads
      call looptest_cuda(type_arr(i))
   enddo
!$omp end parallel do

end program fmain
