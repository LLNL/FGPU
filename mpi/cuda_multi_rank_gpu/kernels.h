#include "cuda_runtime_api.h"

// In saxpy.cu
void testSaxpy_cudac(int rankspergpu);

inline int print_mem(const char *label)
{
  size_t free_on_gpu=0;
  size_t total_on_gpu=0;
  double gbtotal, gbfree, gbused;

  if (cudaMemGetInfo (&free_on_gpu, &total_on_gpu) != cudaSuccess)
  {
      printf ("cudeMemGetInfo failed for GPU 0");
      return (1);
  }

  gbtotal= ((double)total_on_gpu)/(1024.0*1024.0*1024.0);
  gbfree = ((double)free_on_gpu)/(1024.0*1024.0*1024.0);
  gbused = ((double)total_on_gpu-free_on_gpu)/(1024.0*1024.0*1024.0);
  fprintf (stderr, "%s: total %7.4f GB; free %7.3f GB; used %7.3f GB\n", label, gbtotal, gbfree, gbused);
  fflush(stderr);
  return (0);
}
