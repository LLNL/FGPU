program fmain

   use derived_type_mod
   use omp45_kernel_mod

   integer OMP_GET_MAX_THREADS
   integer n, num_sets

   num_sets = OMP_GET_MAX_THREADS()
   print *,"Num sets: ", num_sets

   call construct_types(num_sets)

!$omp parallel do
   do n=1, num_sets
      call alloc_on_device(derived_type_arr(n))
      call omp45_kernel_test(n)
      call destroy_on_device(derived_type_arr(n))
   end do
!$omp end parallel do

end program fmain
