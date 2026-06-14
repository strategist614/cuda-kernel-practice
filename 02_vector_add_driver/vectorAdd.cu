// 这里的 extern C 主要是函数名匹配的问题 要用C语言的方式保留函数名
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        C[i] = A[i] + B[i];
}