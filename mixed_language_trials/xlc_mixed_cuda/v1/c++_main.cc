#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern "C" void fsubroutine(int *);

int main(int argc, char *argv[])
{
   int  numtasks, rank, rc;
   MPI_Comm comm;
   MPI_Fint fcomm;

   rc = MPI_Init(&argc,&argv);
   if (rc != MPI_SUCCESS)
   {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
   }

   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);

   std::cout << "C++: Size " << numtasks << ", Rank " << rank << std::endl;

   comm = MPI_COMM_WORLD;
   fcomm = MPI_Comm_c2f(comm);

   fsubroutine(&fcomm); 

   MPI_Finalize();
	return (0);
}
