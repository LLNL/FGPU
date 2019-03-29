#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{

   int numThreads = omp_get_max_threads();

   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testSaxpy_cudac();
//      testsaxpy_cudafortran();
      testsaxpy_omp45();
   }
	return (0);
}
