#include <stddef.h> // for NULL
#include <signal.h>
 
void xl__trce(int, siginfo_t *, void *);
 
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
	register_xl_sigtrap_handler();
	// ...
}
