#include <cstdio>
#include <iostream>
#include "omp.h"
#include "kernels.h"
#include "cuda_runtime_api.h"

extern "C" void __xlcuf_init(void);

int main(int argc, char *argv[])
{
   int num_iterations = 3;
   size_t free = 0, total = 0;                                                                                                                                                                                     
   cudaError_t status;                                                                                                                                                                                             
 
   std::cout << "Number of devices: " << omp_get_num_devices() << std::endl;
   std::cout << "Default device: " << omp_get_default_device() << std::endl;
   std::cout << "Note: Be sure to set default device via OMP_DEFAULT_DEVICE" << std::endl;
  
   status = cudaMemGetInfo(&free, &total);
   printf("Before initial omp_pause_resource call: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   omp_pause_resource(omp_pause_hard, omp_get_default_device());
	status = cudaMemGetInfo(&free, &total);
  	printf("[After initial omp_pause_resource call]: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   __xlcuf_init();

   printf("after xlcuf_init call: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   for (int i = 0; i < num_iterations; ++i)
   {
   	testSaxpy_cudac();
   	status = cudaMemGetInfo(&free, &total);
   	printf("[After CUDA C kernel]: GPU's memory: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   	testdaxpy_omp45();
   	status = cudaMemGetInfo(&free, &total);
   	printf("[After OpenMP C kernel run]: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

      omp_pause_resource(omp_pause_hard, omp_get_default_device());
   	status = cudaMemGetInfo(&free, &total);
   	printf("[After omp_pause_resource call]: %.2f MB used, %.2f MB free.\n", (double)(total-free)/1048576.0, (double)free/1048576.0);

   }
	return (0);
}
