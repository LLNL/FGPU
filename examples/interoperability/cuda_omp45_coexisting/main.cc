#include <iostream>
#include "omp.h"
#include "kernels.h"

extern "C" void __xlcuf_init(void);

int main(int argc, char *argv[])
{

   __xlcuf_init();

   int numThreads = omp_get_max_threads();
   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {
   	testSaxpy_cudac();
	   testsaxpy_cudafortran();
   	testsaxpy_omp45();
   }
	return (0);
}
