#include <stddef.h> // for NULL
#include <signal.h>
#include <iostream>
#include "omp.h"
 
extern "C" void sub1_f();

extern "C" void xl__trce(int, siginfo_t *, void *);
 
void register_xl_sigtrap_handler()
{
   struct sigaction sa;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sa.sa_sigaction = xl__trce;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGTRAP, &sa, NULL);
}

int main()
{
   int nThreads = 1;
	register_xl_sigtrap_handler();

   std::cout << "Num threads: " << nThreads << std::endl;

   #pragma omp parallel for num_threads(nThreads)
   for (int i = 0; i < nThreads; ++i)
   {
      sub1_f();
   }

	return (0);
}
