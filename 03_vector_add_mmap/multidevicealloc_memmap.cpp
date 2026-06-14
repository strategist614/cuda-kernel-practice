#include "multidevicealloc_memmap.hpp"

static size_t round_up(size_t x, size_t y) { return ((x + y - 1) / y) * y; }

CUresult simpleMallocMultiDeviceMmap(CUdeviceptr                 *dptr,
                                     size_t                      *allocationSize,
                                     size_t                       size,
                                     const std::vector<CUdevice> &residentDevices,
                                     const std::vector<CUdevice> &mappingDevices,
                                     size_t                       align)
{
    CUresult status          = CUDA_SUCCESS;
    size_t   min_granularity = 0;
    size_t   stripeSize;

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

    for (int idx = 0; idx < residentDevices.size(); idx++) {
        size_t granularity = 0;

        prop.location.id = residentDevices[idx];
        status           = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }

    for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
        size_t granularity = 0;

        prop.location.id = mappingDevices[idx];
        status           = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }
    size       = round_up(size, residentDevices.size() * min_granularity);
    stripeSize = size / residentDevices.size();

    if (allocationSize) {
        *allocationSize = size;
    }

    status = cuMemAddressReserve(dptr, size, align, 0, 0);
    if (status != CUDA_SUCCESS) {
        goto done;
    }

    for (size_t idx = 0; idx < residentDevices.size(); idx++) {
        CUresult status2 = CUDA_SUCCESS;

        prop.location.id = residentDevices[idx];

        CUmemGenericAllocationHandle allocationHandle;
        status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
        if (status != CUDA_SUCCESS) {
            goto done;
        }

        status = cuMemMap(*dptr + (stripeSize * idx), stripeSize, 0, allocationHandle, 0);

        status2 = cuMemRelease(allocationHandle);
        if (status == CUDA_SUCCESS) {
            status = status2;
        }

        if (status != CUDA_SUCCESS) {
            goto done;
        }
    }

    {
        std::vector<CUmemAccessDesc> accessDescriptors;
        accessDescriptors.resize(mappingDevices.size());

        for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
           
            accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptors[idx].location.id   = mappingDevices[idx];

            accessDescriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }

        status = cuMemSetAccess(*dptr, size, &accessDescriptors[0], accessDescriptors.size());
        if (status != CUDA_SUCCESS) {
            goto done;
        }
    }

done:
    if (status != CUDA_SUCCESS) {
        if (*dptr) {
            simpleFreeMultiDeviceMmap(*dptr, size);
        }
    }

    return status;
}

CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size)
{
    CUresult status = CUDA_SUCCESS;

    status = cuMemUnmap(dptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    status = cuMemAddressFree(dptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    return status;
}