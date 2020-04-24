#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"
#include "nvToolsExt.h"
#include "cuda_runtime_api.h"
#include "macros.h"

#define USE_CUSTOM_MAP

void testsaxpy_omp45_c()
{
	const int N = 1000000;
   double a = 2.0;

	double *x, *y, *d_x, *d_y;

PUSH_RANGE("C_TEST_KERNEL",1);
   #pragma omp target enter data map(to:a,N)

   x = (double*)malloc(N*sizeof(double));
   y = (double*)malloc(N*sizeof(double));
  
   omp_set_default_device(0);

#if defined(USE_CUSTOM_MAP)
PUSH_RANGE("C_target_alloc",1);
   d_x = (double*) omp_target_alloc( sizeof(double)*N, omp_get_default_device());
   d_y = (double*) omp_target_alloc( sizeof(double)*N, omp_get_default_device());
POP_RANGE();

PUSH_RANGE("C_assoc_ptr",2);
   omp_target_associate_ptr( (void*)x, (void*)d_x, sizeof(double)*N, 0, omp_get_default_device() );
   omp_target_associate_ptr( (void*)y, (void*)d_y, sizeof(double)*N, 0, omp_get_default_device() );
POP_RANGE();

#else
PUSH_RANGE("C_target_map_alloc",1);
   #pragma omp target enter data map(alloc:x[0:N], y[0:N])
POP_RANGE();

#endif

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0;
		y[i] = 2.0;
	}

PUSH_RANGE("C_target_update_to",3);
   #pragma omp target update to(x[0:N],y[0,N])
POP_RANGE();

// Clear y on host
	for (int i = 0; i < N; i++)
	{
		y[i] = 0.0;
   }

PUSH_RANGE("C_daxpy_kernel",4);
   #pragma omp target teams distribute parallel for shared(y,a,x,N) default(none)
   for (int i = 0; i < N; ++i)
   {
      y[i] = a*x[i] + y[i];
   }
POP_RANGE();

PUSH_RANGE("C_target_update_from",5);
   #pragma omp target update from(y[0,N])
POP_RANGE();

#if defined(USE_CUSTOM_MAP)
PUSH_RANGE("C_disassoc_ptr",6);
   omp_target_disassociate_ptr( (void*)x, omp_get_default_device() );
   omp_target_disassociate_ptr( (void*)y, omp_get_default_device() );
POP_RANGE();

PUSH_RANGE("C_target_free",7);
   omp_target_free( (void*)d_x, omp_get_default_device());
   omp_target_free( (void*)d_y, omp_get_default_device());
POP_RANGE();
#else
	#pragma omp target exit data map(delete:x[0:N],y[0:N])
#endif

   #pragma omp target exit data map(delete:a,N)

	double maxError = 0.0;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i]-4.0));
	}
	
	printf(" Ran C OMP45 kernel.  Max error: %f\n", maxError);
	
	free(x);
	free(y);
POP_RANGE();
}
