#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"

void testsaxpy_omp45()
{
	int N = 1<<21;
   int a = 2.0f;

	float *x, *y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float)); 
  
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
 	
   #pragma omp target map(to:N,a,x[:N]) map(y[:N])
   {

   #pragma omp parallel for
   for (int i = 0; i < N; ++i)
   {
      y[i] = a*x[i] + y[i];
		//addFourTimes( i, y );
   }

   }
	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i]-4.0f));
	}
	
	printf(" Ran C OMP45 kernel.  Max error: %f\n", maxError);
	
	free(x);
	free(y);

}
