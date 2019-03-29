   subroutine fsubroutine()

   use looptest_omp45_mod
   use looptest_cuda_mod

   integer OMP_GET_MAX_THREADS
   integer num_max_threads

   print *, "fmain: querying number of threads..."
   num_max_threads = OMP_GET_MAX_THREADS()
   print *, "fmain: number of max threads: ", num_max_threads

!   !$omp parallel
!   call looptest_cuda
!  !$omp end parallel

!  !$omp parallel
   call looptest_omp45
!  !$omp end parallel

   end subroutine fsubroutine
