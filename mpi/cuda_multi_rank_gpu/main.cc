#include "mpi.h"
#include <cstdio>
#include <iostream>
#include "kernels.h"

int main(int argc, char *argv[])
{
   MPI_Init(NULL, NULL);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << " MPI ranks: " << world_size << std::endl << std::flush;
	MPI_Barrier(MPI_COMM_WORLD);

   int gpus = 4;
   int rankspergpu = world_size / gpus;
   size_t free = 0, total = 0;                                                                                                                                                                                     
   cudaError_t status;                                                                                                                                                                                             
 
   MPI_Barrier(MPI_COMM_WORLD);
   print_mem("before cuda c kernel");
   MPI_Barrier(MPI_COMM_WORLD);
   testSaxpy_cudac(rankspergpu);
   MPI_Barrier(MPI_COMM_WORLD);
   print_mem("after cuda c kernel");

	return (0);
}
