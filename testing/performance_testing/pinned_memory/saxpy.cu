#include <stdio.h>
#include "kernels.h"
#include <sys/mman.h>

__global__
void saxpy_cudac(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void testSaxpy_cudac(void)
{
  int N = 1<<21;
  double *x, *y, *d_x, *d_y;


  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));

  mlockall(MCL_CURRENT);
  mlock(x, N*sizeof(double));
  mlock(y, N*sizeof(double));

  cudaMalloc(&d_x, N*sizeof(double)); 
  cudaMalloc(&d_y, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy_cudac<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0));
  printf(" Ran CUDA C kernel.  Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
