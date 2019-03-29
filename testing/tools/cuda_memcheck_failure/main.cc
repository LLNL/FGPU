#include <iostream>
#include "omp.h"
#include "kernels.h"

#if defined(CUDAF)
extern "C" void __xlcuf_init();
#endif

int main(int argc, char *argv[])
{

   int nThreads = 10;

   std::cout << "Num threads: " << nThreads << std::endl;

//   #pragma omp parallel for num_threads(nThreads)
//   for (int i = 0; i < nThreads; ++i)
//   {
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
//   }

	return (0);
}
