#include <cstdio>
#include <iostream>
#include "omp.h"
#include "kernels.h"
#include "cuda_runtime_api.h"

#ifndef USING_CCE
extern "C" void __xlcuf_init(void);
#endif

int main(int argc, char *argv[])
{
   size_t free = 0, total = 0;                                                                                                                                                                                     
   cudaError_t status;                                                                                                                                                                                             
   
   status = cudaMemGetInfo(&free, &total);
   printf("Before xlcuf_init call: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

#ifndef USING_CCE
   __xlcuf_init();
#endif

   int numThreads = omp_get_max_threads();
   std::cout << "Num max threads: " << numThreads << std::endl;

   #pragma omp parallel for
   for (int i = 0; i < numThreads; ++i)
   {

   	testSaxpy_cudac();
   	status = cudaMemGetInfo(&free, &total);
   	printf("After CUDA C kernel run: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);
/*
	  testsaxpy_cudafortran();
   	status = cudaMemGetInfo(&free, &total);
   	printf("After CUDA FORTRAN kernel run: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);
*/
   	testsaxpy_omp45();
   	status = cudaMemGetInfo(&free, &total);
   	printf("After OpenMP C kernel: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   }
	return (0);
}
