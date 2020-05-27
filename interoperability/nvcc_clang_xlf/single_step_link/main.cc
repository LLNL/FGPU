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
      testsaxpy_cudac();
      testsaxpy_omp45_c();
      testsaxpy_omp45_f_();
   }

	return (0);
}
