
      subroutine fhello(comm)
         include 'mpif.h'
         integer, intent(in) :: comm

         integer numtasks, rank, ierr, rc


      write (*,*) "Hello from FORTRAN"
         call MPI_COMM_RANK(comm, rank, ierr)
         call MPI_COMM_SIZE(comm, numtasks, ierr)
         print *, 'Number of tasks=',numtasks,' My rank=',rank


      end
