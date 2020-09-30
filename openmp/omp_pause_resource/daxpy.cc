#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "kernels.h"

void testdaxpy_omp45()
{
	size_t N = 12* 1<<26; // About 12GB
   int a = 2.0f;

   double *x_ptr, *y_ptr;
   x_ptr = (double*) malloc( N*sizeof(double) );
   y_ptr = (double*) malloc( N*sizeof(double) );

   for (int i = 0; i < N; i++)
   {
   	x_ptr[i] = 1.0;
      y_ptr[i] = 2.0;
   }
   print_mem("before mapping N,a,x_ptr[:N], y_ptr[:N]");
   
   #pragma omp target enter data map(to:N,a,x_ptr[:N], y_ptr[:N])

   #pragma omp target teams distribute parallel for
   for (int i = 0; i < N; ++i)
   {
      y_ptr[i] = a*x_ptr[i] + y_ptr[i];
   }

   print_mem("After mapping N,a,x_ptr[:N], y_ptr[:N]");
   #pragma omp target exit data map(from:N,a,x_ptr[:N], y_ptr[:N])
   //#pragma omp target exit data map(from:N,a)

   print_mem("after unmapping x_ptr[:N], y_ptr[:N]");
   
   double maxError = 0.0f;
   for (int i = 0; i < N; i++)
   {
      maxError = fmax(maxError, fabs(y_ptr[i]-4.0f));
   }

   free(x_ptr);
   free(y_ptr);
}
