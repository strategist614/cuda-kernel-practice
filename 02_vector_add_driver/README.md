## CUDA Driver API 向量加法
`.fatbin` 可以理解为已经编译好的 `GPU` 代码文件

* `Driver API`的典型流程
```C++
加载 fatbin / cubin / ptx
↓
从里面找到 kernel 函数
↓
启动 kernel
```

* `Driver API`初始化函数
```C++
checkCudaErrors(cuInit(0));
```
* 需要修改 `makefile` 文件 来适应不同的 GPU
