#include <cstring>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
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
CUdeviceptr d_A = 0, d_B = 0, d_C = 0; // GPU 设备端内存指针
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
        int capable = 0; // 当前 GPU 能不能 peer access 另一个 GPU
        int attributeVal = 0; // 判断另一个 GPU 是否支持 VMM

        if(dev == cuDevice){
            continue;
        }
        // 当前 GPU cuDevice 能不能访问另一个 GPU dev 的显存
        checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
        if(!capable){
            continue;
        }
        // 检查另一个 GPU 是否支持 CUDA 虚拟地址管理
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

    int attributeVal = 0; // 保存设备属性查询结果
    checkCudaErrors(cuInit(0)); // 初始化 CUDA Driver API

    cuDevice = findCudaDeviceDRV(argc, (const char **)argv); // 选择 CUDA 设备 如果是命令行指定 就会选择指定的 如果不指定 会自动选择一个合适的 GPU
    // 检查当前GPU是否支持 VMM
    checkCudaErrors(cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));
    printf("Device %d VIRTUAL ADDRESS MANAGEMENT SUPPORTED: %d\n", cuDevice, attributeVal);
    if(attributeVal == 0){
         printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n", cuDevice);
        exit(EXIT_FAILURE);
    }
    // 这里的 mppingDevices 表示分配出来的虚拟地址允许哪些GPU访问
    vector<CUdevice> mappingDevices;
    mappingDevices.push_back(cuDevice); // 这块内存只映射给当前 GPU 访问
    // 物理显存可以由哪些 GPU 提供
    vector<CUdevice> backingDevices = getBackingDevices(cuDevice);
    CUctxCreateParams ctxCreateParams = {};
    // 创建 context 参数结构体 并初始化为空
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
    // 使用 VMM/MMAP 分配GPU内存
    // 给 d_A 分配 GPU 显存 （分配出来的 GPU 虚拟地址保存到 d_A, 实际分配大小, 原本想申请的大小, 物理显存可以来自哪些 GPU, 这段虚拟地址允许哪些 GPU 访问）
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