#include <cmath>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <nvrtc_helper.h>

int main(int argc, char **argv){
    char *cubin, *kernel_file;
    size_t cubinSize;
    kernel_file = sdkFindFilePath("vectorAdd_kernel.cu", argv[0]);
    compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);
    CUmodule module = loadCUBIN(cubin, argc, argv);

    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "vectorAdd"));

    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = reinterpret_cast<float *>(malloc(size));
    float *h_B = reinterpret_cast<float *>(malloc(size));
    float *h_C = reinterpret_cast<float *>(malloc(size));

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    CUdeviceptr d_A;
    checkCudaErrors(cuMemAlloc(&d_A, size));
    CUdeviceptr d_B;
    checkCudaErrors(cuMemAlloc(&d_B, size));
    CUdeviceptr d_C;
    checkCudaErrors(cuMemAlloc(&d_C, size));

    printf("Copy input data from the host memory to the CUDA device\n");
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    void *arr[] = {reinterpret_cast<void *>(&d_A),
                   reinterpret_cast<void *>(&d_B),
                   reinterpret_cast<void *>(&d_C),
                   reinterpret_cast<void *>(&numElements)};
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                   cudaGridSize.x,
                                   cudaGridSize.y,
                                   cudaGridSize.z, /* grid dim */
                                   cudaBlockSize.x,
                                   cudaBlockSize.y,
                                   cudaBlockSize.z, /* block dim */
                                   0,
                                   0,       /* shared mem, stream */
                                   &arr[0], /* arguments */
                                   0));
    checkCudaErrors(cuCtxSynchronize());

    printf("Copy output data from the CUDA device to the host memory\n");
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}