#include <stdio.h>
#include "kernels.h"

__global__
void saxpy_cudac(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void testsaxpy_cudac(int n, float a, float *x, float *y)
{
  // Perform SAXPY on 1M elements
  saxpy_cudac<<<(n+255)/256, 256>>>(n, a, x, y);
  cudaDeviceSynchronize();
}
