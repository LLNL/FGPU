#include "mpi.h"
#include <cstdio>
#include <iostream>
#include "omp.h"
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
 
	MPI_Barrier(MPI_COMM_WORLD);
   std::cout << " Number of devices: " << omp_get_num_devices() << std::endl << std::flush;
	MPI_Barrier(MPI_COMM_WORLD);
   std::cout << " Default device: " << omp_get_default_device() << std::endl << std::flush;
	MPI_Barrier(MPI_COMM_WORLD);
   print_mem("before hello world");
   MPI_Barrier(MPI_COMM_WORLD);

#pragma omp target
{
   printf("Hello World\n");
}
   MPI_Barrier(MPI_COMM_WORLD);
   print_mem("after omp hello world kernel");
   MPI_Barrier(MPI_COMM_WORLD);
  	testdaxpy_omp45(rankspergpu);
   MPI_Barrier(MPI_COMM_WORLD);
   print_mem("after openmp c kernel");

	return (0);
}
