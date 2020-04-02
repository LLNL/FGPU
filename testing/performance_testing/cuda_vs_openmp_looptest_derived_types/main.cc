#include <iostream>
#include "omp.h"

extern "C" void fsubroutine();

int main(int argc, char *argv[])
{
   omp_set_dynamic(0);     // Explicitly disable dynamic teams
//   omp_set_num_threads(1); // Use 16 threads for all consecutive parallel regions

   int numThreads = omp_get_max_threads();
   std::cout << "cmain: number of max threads: " << numThreads << std::endl;

	fsubroutine();

	return (0);
}
