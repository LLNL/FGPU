#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern "C" void fsubroutine();

int main(int argc, char *argv[])
{
   std::cout << "Hello from C++." << std::endl;
   fsubroutine(); 
	return (0);
}
