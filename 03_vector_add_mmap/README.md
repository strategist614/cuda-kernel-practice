## VMM/MMAP技术

`GPU`虚拟地址和物理显存映射的一套机制

普通的显存分配是：
```C++
cuMemAlloc(&d_A, size);
```

`VMM/MMAP`是更底层的方式:
```C
保留一段 GPU 虚拟地址
        ↓
创建一块真实的 GPU 物理显存
        ↓
把物理显存映射到虚拟地址上
        ↓
设置哪些 GPU 可以访问这段地址
```

`VMM/MMAP`：
```C
GPU 虚拟地址管理 + GPU 物理显存映射
```

代码中的 `simpleMallocMultiDeviceMmap`是：
```C++
cuMemAddressReserve
cuMemCreate
cuMemMap
cuMemSetAccess
```

`allocationSize`是真实分配大小
释放时需要实际的大小

`VMM/MMAP`的用处
* 多 `GPU` 显存管理
* 虚拟地址连续，但物理显存可以不连续，可以保留一大段连续虚拟地址，但是背后的物理显存可以分段映射。
* 稀疏映射，先保留很大的虚拟地址空间，但只给其中一部分真正映射物理显存。
* 动态 `remap`，把同一段虚拟地址背后的物理显存换掉。