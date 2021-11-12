#include <cstdio>
#include <iostream>
#include "omp.h"
#include "kernels.h"

extern "C" void __xlcuf_init(void);

int main(int argc, char *argv[])
{
   int num_iterations = 3;
   size_t free = 0, total = 0;                                                                                                                                                                                     
   cudaError_t status;                                                                                                                                                                                             
 
   std::cout << "Number of devices: " << omp_get_num_devices() << std::endl;
   std::cout << "Default device: " << omp_get_default_device() << std::endl;
   std::cout << "Note: Be sure to set default device via OMP_DEFAULT_DEVICE" << std::endl;

   print_mem("before initial omp_pause_resource");
   omp_pause_resource(omp_pause_hard, omp_get_default_device());
   print_mem("after initial omp_pause_resource");

   __xlcuf_init();
   print_mem("after xlcuf_init");

   for (int i = 0; i < num_iterations; ++i)
   {
   	testSaxpy_cudac();
      print_mem("after cuda c kernel");

   	testdaxpy_omp45();
      print_mem("after openmp c kernel");

      omp_pause_resource(omp_pause_hard, omp_get_default_device());
      print_mem("after omp pause resource");
   }
	return (0);
}
