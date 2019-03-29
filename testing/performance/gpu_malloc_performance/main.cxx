#include <iostream>
#include "omp.h"
#include <cmath>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{

   unsigned long int n0=pow(2, 2);
   unsigned long int n1=pow(2,33);
   unsigned long int i;

   cudaError_t err;

   void *x_omp;
   void *x_cuda;
   double wtime;

   omp_set_default_device(0);

   for (i = n0; i <= n1;)
   {
      wtime = omp_get_wtime();
      x_omp = omp_target_alloc( i, omp_get_default_device());
      if (x_omp == nullptr)
		{
			std::cerr << "Failed omp allocation of " << i/1024.0/1024.0 << " mbytes." << std::endl;
		}
      wtime = omp_get_wtime() - wtime;
      std::cout << std::fixed << wtime << ": target alloc " << i/1024.0/1024.0 << " mbytes:" << std::endl;

      wtime = omp_get_wtime();
      omp_target_free( x_omp, omp_get_default_device());
      wtime = omp_get_wtime() - wtime;
      std::cout << std::fixed << wtime << ": target free " << i/1024.0/1024.0 << " mbytes:" << std::endl;

      i = i*2;
	}
	
   i = pow(2,34) - 1024*1024*1024;
   wtime = omp_get_wtime();
   x_omp = omp_target_alloc( i, omp_get_default_device());
   if (x_omp == nullptr)
	{
		std::cerr << "Failed allocation of " << i/1024.0/1024.0 << " mbytes." << std::endl;
	}
   wtime = omp_get_wtime() - wtime;
   std::cout << std::fixed << wtime << ": target alloc " << i/1024.0/1024.0 << " mbytes:" << std::endl;

   wtime = omp_get_wtime();
   omp_target_free( x_omp, omp_get_default_device());
   wtime = omp_get_wtime() - wtime;
   std::cout << std::fixed << wtime << ": target free " << i/1024.0/1024.0 << " mbytes:" << std::endl;

	std::cout << "----------------------------------------" << std::endl;

#if 0
	for (i = n0; i <= n1;)
	{
      wtime = omp_get_wtime();
      err = cudaMalloc(&x_cuda, i );
      if (err != cudaSuccess)
		{
			std::cerr << "Failed cuda allocation of " << i/1024.0/1024.0 << " mbytes." << std::endl;
		}
      wtime = omp_get_wtime() - wtime;
      std::cout << std::fixed << wtime << ": target alloc " << i/1024.0/1024.0 << " mbytes:" << std::endl;

      err = cudaFree(x_cuda);

      i = i*2;
	}
	
   i = pow(2,34) - 1024*1024*1024;
   wtime = omp_get_wtime();
   err = cudaMalloc(&x_cuda, i);
   if (err != cudaSuccess)
   {
		std::cerr << "Failed cuda allocation of " << i/1024.0/1024.0 << " mbytes." << std::endl;
	}
   wtime = omp_get_wtime() - wtime;
   std::cout << std::fixed << wtime << ": target alloc " << i/1024.0/1024.0 << " mbytes:" << std::endl;
   err = cudaFree(x_cuda);
#endif
	return (0);
}
