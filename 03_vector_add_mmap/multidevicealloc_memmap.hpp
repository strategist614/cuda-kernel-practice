#pragma once
#include <cuda.h>
#include <vector>

CUresult simpleMallocMultiDeviceMmap(CUdeviceptr                 *dptr,
                                     size_t                      *allocationSize,
                                     size_t                       size,
                                     const std::vector<CUdevice> &residentDevices,
                                     const std::vector<CUdevice> &mappingDevices,
                                     size_t                       align = 0);
CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size);