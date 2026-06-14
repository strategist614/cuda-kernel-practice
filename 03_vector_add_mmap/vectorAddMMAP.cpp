#include <cstring>
#include <cuda.h>
#include <iostream>
#inlcude <stdio.h>
#inlcude <string.h>

#include <helper_cuda_drvapi.h>
#include < helper_functions.h>




int CleanupNoFailure()
{
    // Free device memory
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSize));

    // Free host memory
    if (h_A) {
        free(h_A);
    }

    if (h_B) {
        free(h_B);
    }

    if (h_C) {
        free(h_C);
    }

    checkCudaErrors(cuModuleUnload(cuModule));
    checkCudaErrors(cuCtxDestroy(cuContext));

    return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float *data, int n)
{
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}