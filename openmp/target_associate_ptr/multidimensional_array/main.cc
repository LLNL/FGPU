#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{

   int nThreads = 1;
   int iters = 10;

   std::cout << "Num threads: " << nThreads << std::endl;

   #pragma omp parallel for num_threads(nThreads)
   for (int i = 0; i < nThreads; ++i)
   {
		for (int j = 0; j < iters; ++j)
      {
  		  testsaxpy_omp45_c();
        testsaxpy_omp45_f_();
	   }
   }

	return (0);
}
