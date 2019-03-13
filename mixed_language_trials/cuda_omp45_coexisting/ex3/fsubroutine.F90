   subroutine fsubroutine()

   use looptest1_omp45_mod
   use looptest2_omp45_mod
   use looptest_cuda_mod

   integer num_max_threads
   integer OMP_GET_MAX_THREADS

	integer n

   num_max_threads = OMP_GET_MAX_THREADS()
   print *, "fmain: number of max threads: ", num_max_threads

	!$omp parallel
   call looptest_cuda()
   !$omp end parallel

   !call looptest1_omp45()
   !call looptest2_omp45()

   end subroutine fsubroutine
