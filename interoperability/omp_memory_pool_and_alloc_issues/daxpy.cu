#include <stdio.h>
#include "kernels.h"
#include "omp.h"
#include "cuda_runtime_api.h"

__global__
void daxpy_cudac(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void testDaxpy_cudac(void)
{
  size_t N = 12*(1<<26); // About 12GB
  double *x, *y, *d_x, *d_y;
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t status;                                                                                                                                                                                             
   
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));

  d_x = (double*)omp_target_alloc(N*sizeof(double), omp_get_default_device());
  d_y = (double*)omp_target_alloc(N*sizeof(double), omp_get_default_device());

//  cudaMalloc(&d_x, N*sizeof(double)); 
//  cudaMalloc(&d_y, N*sizeof(double));

  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("In CUDA C kernel: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total_bytes-free_bytes)/1048576.0, (double)free_bytes/1048576.0);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  daxpy_cudac<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0));
  printf("-- Ran CUDA C kernel.  Max error: %f\n", maxError);

//  cudaFree(d_x);
//  cudaFree(d_y);

  omp_target_free(d_x, omp_get_default_device());
  omp_target_free(d_y, omp_get_default_device());

  free(x);
  free(y);
}
