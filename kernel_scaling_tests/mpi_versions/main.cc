#include <iostream>
#include <mpi.h>
#include "omp.h"
#include "kernels.h"

#if defined(CUDAF)
extern "C" void __xlcuf_init();
#endif

int main(int argc, char *argv[])
{

   MPI_Init(NULL,NULL);

#if !defined(PGI)
   __xlcuf_init();
#endif

   int numThreads = omp_get_max_threads();

   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
		#if defined (CUDAC)
      testsaxpy_cudac();
		#endif

		#if defined(OMP45C)
      testsaxpy_omp45_c();
		#endif

      #if defined(OMP45F)
      testsaxpy_omp45_f_();
      #endif

      #if defined(CUDAF)
      testsaxpy_cudafortran_();
      #endif
   }
   MPI_Finalize();

	return (0);
}
