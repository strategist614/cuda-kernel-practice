### VectorAdd
* cudaMalloc 是在 GPU 中分配显存的
* blockIdx.x 表示block在整个grid的编号
* blockDim.x 表示每个block有多少个线程
* blockIdx.x * blockDim.x 表示当前到多少个block
* blockIdx.x * blockDim.x + threadIdx.x 表示偏移了多少个线程
