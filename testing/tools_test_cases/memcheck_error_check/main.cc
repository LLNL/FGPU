#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
   omp_set_dynamic(0);     // Explicitly disable dynamic teams
   omp_set_num_threads(1); // Use 16 threads for all consecutive parallel regions

   int numThreads = omp_get_max_threads();

   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
      testsaxpy_cudafortran();
      testsaxpy_omp45();
   }
	return (0);
}
