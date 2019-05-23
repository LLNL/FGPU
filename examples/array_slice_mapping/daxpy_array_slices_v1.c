// This example attempts to map over slices of an array to a GPU, then launch kernels that act on those slices.
#include "driver_types.h" // for cudaError_t
#include "cuda_runtime_api.h" // for cudaHostAlloc
#include <stdio.h> // for printf

int main()
{
   int num_slices, num_values, a;
	cudaError_t err;

	double foo;
   double *x, *y;

   a = 5;
   num_slices = 80;
   num_values = 1024; // Make at least 10, since the first 10 values are printed later.

	err = cudaHostAlloc((void**)&x, sizeof(foo) * num_values * num_slices, cudaHostAllocDefault );
	err = cudaHostAlloc((void**)&y, sizeof(foo) * num_values * num_slices, cudaHostAllocDefault );

   // initialize arrays
   for (int j=0; j < num_slices; ++j)
	{
		for (int i=0; i < num_values; ++i)
		{
      	x[j*num_values+i] = (double)j;
      	y[j*num_values+i] = (double)j;
	   }
	}
   for (int j=0; j < num_slices; ++j)
	{
     	printf("----------\n");
      printf("before x[1:10,%d] ",j);

		for (int i=0; i < 10; ++i)
		{
			printf("%f ", x[j*num_values+i]);
		}
     	printf("\n----------\n");
	}

   for (int j=0; j < num_slices; ++j)
	{
      #pragma openmp target enter data map(to:x(j*num_slices:j*num_slices+num_values))
      printf("Mapped x(:,%d)\n", j);
      #pragma openmp  target enter data map(to:y[j*num_slices:j*num_slices+num_values])
      printf("Mapped y(:,%d)\n", j);
	}

   for (int j=0; j < num_slices; ++j)
	{
      #pragma openmp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) default(none)
      for (int i=0; i < num_values; ++i)
		{
         x[j*num_values+i] = a * x[j*num_values+i] + y[j*num_values+i];
		}
      #pragma openmp end target teams distribute parallel do

      printf("Ran daxpy on slice %d\n", j);
	}

   for (int j=0; j < num_slices ; ++j)
	{
      #pragma openmp target exit data map(from:x(j*num_slices:j*num_slices+num_values))
      #pragma openmp target exit data map(delete:y(j*num_slices:j*num_slices+num_values))
   }

   for (int j=0; j < num_slices; ++j)
	{
     	printf("----------\n");
      printf("after x[1:10,%d] ",j);

		for (int i=0; i < 10; ++i)
		{
			printf("%f ", x[j*num_values+i]);
		}
     	printf("\n----------\n");
	}

	return 0;
}
