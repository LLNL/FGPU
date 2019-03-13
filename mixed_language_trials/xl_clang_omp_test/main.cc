#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{

   int nThreads = 10;
   int i;

   std::cout << "Num threads: " << nThreads << std::endl;

	#pragma omp parallel for
	for (i=0; i<10; ++i)
   { 
     #if defined(OMP45C)
     testsaxpy_omp45_c();
  	  #endif

     #if defined(OMP45F)
     testsaxpy_omp45_f_();
     #endif
   }

	return (0);
}
