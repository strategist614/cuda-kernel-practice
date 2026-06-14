#include <cstring>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

#include <builtin_types.h>

// 这里封装了 VMM/MMAP 显存分配函数 simpleMallocMultiDeviceMmap 和 simpleFreeMultiDeviceMmap
#include "multidevicealloc_memmap.hpp"

using namespace std;

CUdevice cuDevice; // 一个CUDA设备
CUcontext cuContext; // CUDA上下文，CUDA的资源分配和管理都在上下文中进行
CUmodule cuModule; // 加载进来的 CUDA 模块，.ptx .fatbin 加载之后形成的模块对象
CUfunction vecAdd_kernel; // 从 CUDA module 里面拿到的 kernel 函数

float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
CUdeviceptr d_A, d_B, d_C; // GPU 设备端内存指针
size_t allocationSize = 0; // 实际分配显存的大小
// VMM/MMAP 分配时，实际分配大小通常会按照 GPU 内存粒度对齐
int CleanupNoFailure();
void RandomInit(float *data, int n);
#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

// 找出哪些GPU可以作为 backing devices
// backing device 表示物理显存可以来自哪些GPU
vector<CUdevice> getBackingDevices(CUdevice cuDevice){
    int num_devices; // 当前机器上的 CUDA GPU 数量

    checkCudaErrors(cuDeviceGetCount(&num_devices)); //获取GPU数量

    vector<CUdevice> backingDevices;
    backingDevices.push_back(cuDevice); // 先把当前的 GPU 加入 backing devices 

    for (int dev = 0; dev < num_devices; dev++){
        int capable = 0;
        int attributeVal = 0;

        if(dev == cuDevice){
            continue;
        }

        checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
        if(!capable){
            continue;
        }

        checkCudaErrors(cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
        if(attributeVal == 0){
            continue;
        }
        backingDevices.push_back(dev);
    }
    return backingDevices;
}

int main(int argc, char **argv)
{
    printf("Vector Addition (Driver API)\n");

    int N = 50000;
    size_t size = N * sizeof(float);

    int attributeVal = 0;
    checkCudaErrors(cuInit(0));

    cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

    checkCudaErrors(cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));
    printf("Device %d VITUAL ADDRESS MANAGEMENT SUPPORTED: %d\n", cuDevice, attributeVal);
    if(attributeVal == 0){
         printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n", cuDevice);
        exit(EXIT_FAILURE);
    }

    vector<CUdevice> mappingDevices;
    mappingDevices.push_back(cuDevice);

    vector<CUdevice> backingDevices = getBackingDevices(cuDevice);
    CUctxCreateParams ctxCreateParams = {};

    checkCudaErrors(cuCtxCreate(&cuContext, &ctxCreateParams, 0, cuDevice));

    string module_path;
    std::ostringstream fatbin;

    if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin)) {
        exit(EXIT_FAILURE);
    }
    else {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    if (!fatbin.str().size()) {
        printf("fatbin file empty. exiting..\n");
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

    checkCudaErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_A, &allocationSize, size, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_B, NULL, size, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_C, NULL, size, backingDevices, mappingDevices));

    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = {&d_A, &d_B, &d_C, &N};

    checkCudaErrors(cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL));

    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

    int i;

    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f) {
            break;
        }
    }

    CleanupNoFailure();
    printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

    exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure()
{
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSize));

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

void RandomInit(float *data, int n)
{
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}