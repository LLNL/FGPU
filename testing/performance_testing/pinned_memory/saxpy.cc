#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"
#include <sys/mman.h>

void testsaxpy_omp45()
{
	int N = 1<<21;
   int a = 2.0f;

	double *x, *y;


	x = (double*)malloc(N*sizeof(double));
	y = (double*)malloc(N*sizeof(double)); 
  
   mlockall(MCL_CURRENT);
	mlock(x, N*sizeof(double));
	mlock(y, N*sizeof(double));

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
   }

   }
	
	double maxError = 0.0;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i]-4.0));
	}
	
	printf(" Ran C OMP45 kernel.  Max error: %f\n", maxError);
	
	free(x);
	free(y);

}
