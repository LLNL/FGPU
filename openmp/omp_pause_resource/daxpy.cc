#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"
#include "cuda_runtime_api.h"

void testdaxpy_omp45()
{
	size_t N = 1<<28;
   int a = 2.0f;
   size_t free_bytes = 0, total_bytes = 0;
   cudaError_t status;

	double *x_ptr, *y_ptr;
	x_ptr = (double*) malloc( N*sizeof(double) );
	y_ptr = (double*) malloc( N*sizeof(double) ); 
  
	for (int i = 0; i < N; i++)
	{
		x_ptr[i] = 1.0;
		y_ptr[i] = 2.0;
	}
 	
   #pragma omp target enter data map(to:N,a,x_ptr[:N], y_ptr[:N])
   status = cudaMemGetInfo(&free_bytes, &total_bytes);
   printf("In C OpenMP kernel run: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total_bytes-free_bytes)/1048576.0, (double)free_bytes/1048576.0);

   #pragma omp target teams distribute parallel for
   for (int i = 0; i < N; ++i)
   {
      y_ptr[i] = a*x_ptr[i] + y_ptr[i];
   }

   #pragma omp target exit data map(release:N,a,x_ptr[:N])
   #pragma omp target exit data map(from:y_ptr[:N])

	double maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y_ptr[i]-4.0f));
	}
	
	printf("-- Ran C OMP45 kernel.  Max error: %f\n", maxError);
	
	free(x_ptr);
	free(y_ptr);

}
