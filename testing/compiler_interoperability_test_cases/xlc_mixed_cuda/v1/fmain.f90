program fmain

   use mpi

   implicit none
   integer ( kind = 4 ) ierr

   call MPI_Init( ierr )

   write (*,*) "Hello."

   call test_saxpy_cuda_kernels

   call MPI_Finalize( ierr );

end program fmain
