#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

int main (void)
{
    unsigned char *p;
    cudaError_t stat;
    size_t siz = 1024ULL * 1024 * 6200;

    printf("allocating %d bytes", siz);
    stat = cudaHostAlloc (&p, siz, cudaHostAllocDefault);
    if (stat == cudaSuccess) {
        volatile unsigned long long count = 1ULL<<34;
        printf ("allocation succesful\n");fflush(stdout);
        do {
            count--;
        } while (count);
        cudaFreeHost(p);
    }
    return EXIT_SUCCESS;
}
