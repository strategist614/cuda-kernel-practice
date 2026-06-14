#include<cstring>
#include<iostream>
#include<cuda.h>
#include<stdio.h>
#include<string.h>

#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

#include<builtin_types.h>
#include <sstream>

using namespace std;

CUdevice cuDevice; // cuDevice 当前使用的GPU
CUcontext cuContext; // cuContext 当前使用的GPU上下文
CUmodule cuModule; // 被加载进来的 CUDA模块（.fatbin文件、.cubin文件和.ptx文件）
CUfunction vecAdd_kernel; // 从模块取出来的 GPU kernel函数
float *h_A, *h_B, *h_C; // CPU内存
CUdeviceptr d_A, d_B, d_C; // GPU显存

int CleanupNoFailure();
void RandomInit(float* , int);
bool findModulePath(const char* , string&, char **, string &);

// 定义了要加载的 fatbin 文件
#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel.fatbin"
#endif

int main(int argc, char **argv){
    printf("Vector Addition (Driver API)\n");
    int N = 50000, devID=0;
    size_t size = N * sizeof(float);
    CUctxCreateParams ctxCreateParams={};
    checkCudaErrors(cuInit(0)); // 初始化 CUDA Driver API
    cuDevice = findCudaDeviceDRV(argc, (const char **)argv); // 找到一个可用的 CUDA 设备
    checkCudaErrors(cuCtxCreate(&cuContext, &ctxCreateParams, 0, cuDevice)); // 创建 CUDA 上下文 为GPU创建一个运行环境
    string module_path; // 保存 fatbin 文件的路径
    std::ostringstream fatbin; // 用来保存 fatbin 文件内容的字符串流对象
    
    // 寻找 vectorAdd_kernel.fatbin 文件路径
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

    checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str())); // 加载 fatbin 文件中的 GPU 代码
    checkCudaErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel")); // 从加载的模块中获取 GPU kernel 函数 VecAdd_kernel 的函数句柄

    // 分配 CPU内存
    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // 分配 GPU显存
    checkCudaErrors(cuMemAlloc(&d_A,size));
    checkCudaErrors(cuMemAlloc(&d_B,size));
    checkCudaErrors(cuMemAlloc(&d_C,size));
    
    // 把数据从 CPU 拷贝到 GPU
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

    if (1) {
        // 设置 kernel 启动参数
        int threadsPerBlock = 256;
        // blocksPerGrid = 196
        int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

        // kernel 启动参数列表，按照 VecAdd_kernel 函数参数顺序依次传入
        void *args[] = {&d_A, &d_B, &d_C, &N};
        // 启动 kernel 
        // 对应 Runtime API 的 kernel 启动方式：VecAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        checkCudaErrors(cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL));
    }
    // else {
    //     int   offset = 0;
    //     char argBuffer[256];
    //     *((CUdeviceptr *)&argBuffer[offset]) = d_A;
    //     offset += sizeof(d_A);
    //     *((CUdeviceptr *)&argBuffer[offset]) = d_B;
    //     offset += sizeof(d_B);
    //     *((CUdeviceptr *)&argBuffer[offset]) = d_C;
    //     offset += sizeof(d_C);
    //     *((int *)&argBuffer[offset]) = N;
    //     offset += sizeof(N);

    //     int threadsPerBlock = 256;
    //     int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    //     checkCudaErrors(
    //         cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, NULL, argBuffer));
    // }
    #ifdef DEBUG
        checkCudaErrors(cuCtxSynchronize());
    #endif

        checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size)); // 把结果从 GPU 拷贝回 CPU
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
    return 0;
}

int CleanupNoFailure()
{
    // Free device memory
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));

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