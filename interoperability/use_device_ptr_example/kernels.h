// In saxpy.cuf
extern "C" void testsaxpy_cudafortran_();

// In saxpy.F90
extern "C" void testsaxpy_omp45_f_();

// In saxpy.cu
void testsaxpy_cudac(int n, float a, float *x, float *y);

// In saxpy.c
void testsaxpy_omp45_c(void);

