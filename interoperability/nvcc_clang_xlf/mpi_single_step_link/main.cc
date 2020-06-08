#include <iostream>
#include "mpi.h"
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{

   int request = MPI_THREAD_MULTIPLE;
   int provided;

   if ( MPI_Init_thread(&argc,&argv,request, &provided) != MPI_SUCCESS )                                                                                                                                
   {                                                                                                                                                                                                    
      exit(1);                                                                                                                                                                                          
   }                                                                                                                                                                                                    

   int numThreads = omp_get_max_threads();

   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_cudac();
      MPI_Barrier(MPI_COMM_WORLD);
      testsaxpy_omp45_c();
      MPI_Barrier(MPI_COMM_WORLD);
      testsaxpy_omp45_f_();
      MPI_Barrier(MPI_COMM_WORLD);
   }
   MPI_Barrier(MPI_COMM_WORLD);

	return (0);
}
