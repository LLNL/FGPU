#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"

void testsaxpy_omp45_c()
{
	int N = 1<<21;
   int a = 2.0f;

	double *x, *y, *d_x, *d_y;

	x = (double*)malloc(N*sizeof(double));
	y = (double*)malloc(N*sizeof(double)); 
  
   omp_set_default_device(0);

   d_x = (double*) omp_target_alloc( sizeof(double)*N, omp_get_default_device());
   d_y = (double*) omp_target_alloc( sizeof(double)*N, omp_get_default_device());

   omp_target_associate_ptr( (void*)x, (void*)d_x, sizeof(double)*N, 0, omp_get_default_device() );
   omp_target_associate_ptr( (void*)y, (void*)d_y, sizeof(double)*N, 0, omp_get_default_device() );

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
 	
   #pragma omp target update to(x[0:N],y[0,N])

   #pragma target omp teams distribute parallel for shared(y,a,x,N) default(none)
   for (int i = 0; i < N; ++i)
   {
      y[i] = a*x[i] + y[i];
   }

   #pragma omp target update from(y[0,N])

   omp_target_disassociate_ptr( (void*)x, omp_get_default_device() );
   omp_target_disassociate_ptr( (void*)y, omp_get_default_device() );
   omp_target_free( (void*)d_x, omp_get_default_device());
   omp_target_free( (void*)d_y, omp_get_default_device());

	double maxError = 0.0;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i]-4.0f));
	}
	
	printf(" Ran C OMP45 kernel.  Max error: %f\n", maxError);
	
	free(x);
	free(y);

}
