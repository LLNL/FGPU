#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern "C" void fsubroutine();

int main(int argc, char *argv[])
{
   csubroutine();
   fsubroutine();
	return (0);
}
