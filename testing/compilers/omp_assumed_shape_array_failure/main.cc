#include <iostream>
#include "omp.h"
#include "kernels.h"

#if defined(CUDAF)
extern "C" void __xlcuf_init()
#endif

int main(int argc, char *argv[])
{

#if defined(CUDAF)
   __xlcuf_init();
#endif

   int numThreads = omp_get_max_threads();

   std::cout << "Num max threads: " << numThreads << std::endl;

#if defined (CUDAC)
   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_cudac();
   }

#endif

#if defined(OMP45C)
   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_omp45_c();
   }

#endif

#if defined(OMP45F)
   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_omp45_f_();
   }
#endif 

#if defined(CUDAF)
   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_cudafortran_();
   }
#endif

	return (0);
}
