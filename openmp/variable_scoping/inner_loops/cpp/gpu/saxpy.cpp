#include <omp.h>
#include <vector>
#include <iostream>


// QUESTION
// Remove the 'private(j,k)'.  Compiler should complain these are not scoped due to the 'default(none)'.
// XL - does not catch this error for the target teams case.  It does catch this if the target team distribute is removed.
// TEST
// Change the 'private(j,k)' to a shared(j,k).  The compiler should complain that loop iteration variables can not be shared.
// XL - Does catch this error.

void saxpy(int n, float a, std::vector<float>& x, std::vector<float>& y)
{
    int i, j, k;

    #pragma omp target teams distribute parallel for default(none) private(j,k) shared(n, a, x, y)
    for (i = 0; i < n; ++i) {
        float temp = a * x[i];
        
        // First inner serial loop
        for (j = 1; j <= 5; ++j) {
            temp += j;
        }
        
        // Second inner serial loop
        for (k = 1; k <= 3; ++k) {
            temp -= k;
        }
        
        y[i] = temp + y[i];
    }
}

int main() {
    int n = 100000; // Size of vectors
    float a = 2.0; // Scalar value
    std::vector<float> x(n, 1.0); // Vector X initialized with 1.0
    std::vector<float> y(n, 2.0); // Vector Y initialized with 2.0

    // Perform SAXPY operation
    saxpy(n, a, x, y);

    // Example: Print the first 10 results
    for (int i = 0; i < 10; ++i) {
        std::cout << y[i] << std::endl;
    }

    return 0;
}
