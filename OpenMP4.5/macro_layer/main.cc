#include <iostream>
#include "omp.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
      testsaxpy_omp45_c();
      testsaxpy_omp45_f_();

	return (0);
}
