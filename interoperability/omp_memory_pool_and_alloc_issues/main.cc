#include <cstdio>
#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
   int num_iterations = 10;
   
   size_t N = 12*(1<<26); // About 6GB?
   double *d_x, *d_y;                                                                                                                                              

   for (int i = 0; i < num_iterations; ++i)
   {

		printf("------ITER %d --------\n",i);

      // 1st package runs on GPU.  Utilizes primarily OpenMP maps.
      printf("\n-- run kernel 1 --\n");
   	testdaxpy_omp45();
		//omp_pause_resource_all(omp_pause_hard);

      // 2nd package does not utiliize OpenMP maps.
      // It makes direct device memory allocation calls on GPU for its needed memory.
      printf("\n-- run kernel 2 (placeholder) --\n");
      d_x = (double*)omp_target_alloc(N*sizeof(double), omp_get_default_device());
      if (d_x == NULL)
      {
         printf("Failed to allocate d_x.\n");
         exit(1);
      }
      d_y = (double*)omp_target_alloc(N*sizeof(double), omp_get_default_device());
      if (d_y == NULL)
      {
         printf("Failed to allocate d_y.\n");
         exit(1);
      }

      // 2nd package runs here
      
      // 2nd package frees memory from GPU
      omp_target_free(d_x, omp_get_default_device());
      omp_target_free(d_y, omp_get_default_device());
		printf("--------------------\n\n");

   }
	return (0);
}
