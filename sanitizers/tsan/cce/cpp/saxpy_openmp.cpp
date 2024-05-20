#include <omp.h>
#include <vector>
#include <iostream>

int main() {
    const int N = 1000000;
    float alpha = 2.0f;
    std::vector<float> X(N, 1.0f); // Initialize X with 1.0
    std::vector<float> Y(N, 2.0f); // Initialize Y with 2.0

    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        Y[i] = alpha * X[i] + Y[i];
    }

    std::cout << "The first element of Y after SAXPY is: " << Y[0] << std::endl;

    return 0;
}
