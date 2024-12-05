#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string.h>
#include <nvtx3/nvToolsExt.h>
// #include <thrust/fill.h>
// #include <thrust/execution_policy.h>

void dataUpdate(char* c_data, size_t GB){
        //update first part 
    for (size_t i = 0; i < 10; ++i) {
        c_data[i] = static_cast<char>(i % 256 + 10); // Example modification
    }

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "c_data[" << i << "] first is h_data = " << static_cast<int>(c_data[i]) << std::endl;
    };
    for (size_t i = sizeof(char) * GB; i < (sizeof(char) * GB + 10); ++i) {
        std::cout << "c_data[" << i << "] after half is d_data = " << static_cast<int>(c_data[i]) << std::endl;
    };

}
void dataCheck(char* c_data, char* d_data, size_t GB, cudaError_t err, int offset){
    //test for result after unmap
    err = cudaMemcpy(c_data+offset, d_data+offset , sizeof(char) * GB, cudaMemcpyDeviceToHost); // check last part
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        free(c_data);
        // cudaFree(d_data);
        return ;
    };
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "c_data[" << i << "] first is h_data = " << static_cast<int>(c_data[i]) << std::endl;
    };
    for (size_t i = sizeof(char) * GB; i < (sizeof(char) * GB + 10); ++i) {
        std::cout << "c_data[" << i << "] after half is d_data = " << static_cast<int>(c_data[i]) << std::endl;
    };

};

int main(){
    //allocate a contiguous 2GB buffer where 1 GB resides on GPU 0 and 1 GB resides on the host
    //API requires cuda 12.2, driver bug is fixed with cuda 12.4 / driver 550.54.14 (linux)
    bool checkflag = true;
    
    // constexpr size_t GB = 1 << 30;
    size_t GB = 1 << 30;
    // GB *= 10;
    cudaSetDevice(0); //initialize cuda context

    CUresult status = CUDA_SUCCESS;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularityDevice = 0;
    size_t granularityHost = 0;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    status = cuMemGetAllocationGranularity(&granularityDevice, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);

    prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id = 0;
    status = cuMemGetAllocationGranularity(&granularityHost, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);

    size_t granularity = std::max(granularityDevice, granularityHost); // need by create GB

    const size_t allocationSize = 2*GB;
    assert(GB % granularity == 0);
    assert(allocationSize % granularity == 0);

    CUdeviceptr deviceptr = 0; // point addr for the vmm
    nvtxRangePushA("init Range");

    // reserve
    status = cuMemAddressReserve(&deviceptr, allocationSize, 0, 0, 0); // reserve allocsize
    assert(status == CUDA_SUCCESS);

    // alloc physical
    CUmemGenericAllocationHandle allocationHandle;
    // prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; // host numa
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // add by 12.4
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    status = cuMemCreate(&allocationHandle, GB, &prop, 0); // physical at allocationHandle
    assert(status == CUDA_SUCCESS);
    // map at the end of deviceptr
    status = cuMemMap(deviceptr, GB, 0, allocationHandle, 0);
    assert(status == CUDA_SUCCESS);
    // do not need handle after map
    // status = cuMemRelease(allocationHandle);
    // assert(status == CUDA_SUCCESS);

    CUmemGenericAllocationHandle allocationHandle_2;  //maybe could be reuse
    // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // add by 12.4
    // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; // host numa
    prop.location.id = 0;
    status = cuMemCreate(&allocationHandle_2, GB, &prop, 0);
    assert(status == CUDA_SUCCESS);
    status = cuMemMap(deviceptr + GB, GB, 0, allocationHandle_2, 0); // alloc the other area within the same vmm
    assert(status == CUDA_SUCCESS);
    // status = cuMemRelease(allocationHandle_2);
    // assert(status == CUDA_SUCCESS);

    // set access 
    //set access control such that the device chunk is only accessible from the device,
    //and the host chunk is also only accessible from the device
    std::vector<CUmemAccessDesc> accessDescriptors(1);
    accessDescriptors[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptors[0].location.id = 0;
    accessDescriptors[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = cuMemSetAccess(deviceptr, GB, accessDescriptors.data(), 1);
    assert(status == CUDA_SUCCESS);
    status = cuMemSetAccess(deviceptr + GB, GB, accessDescriptors.data(), 1);
    assert(status == CUDA_SUCCESS);
    nvtxRangePop();

    // allocate data
    char* d_data = (char*)deviceptr; // addr for the vmm 
    char* h_data; cudaMallocHost(&h_data, sizeof(char) * 2*GB);

    for (size_t i = 0; i < sizeof(char) * 2*GB; ++i) {
        h_data[i] = static_cast<char>(i % 256); // Example modification
    }

    // Print the first few values to verify
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << static_cast<int>(h_data[i]) << std::endl;
    };

    // data for check on device
    char* c_data;
    c_data = (char *)malloc(sizeof(char) * 2*GB);
    if (c_data == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        cudaFree(c_data);
        return -1;
    }

    nvtxRangePushA("cpy Range");
    // older drivers may report errors on the next lines
    cudaError_t rtstatus = cudaSuccess;
    rtstatus = cudaMemset(d_data, 0, GB);
    std::cout << "cudaMemset device chunk: " << cudaGetErrorString(rtstatus) << "\n";
    cudaGetLastError();

    rtstatus = cudaMemset(d_data + GB, 0, GB);
    std::cout << "cudaMemset host chunk: " << cudaGetErrorString(rtstatus) << "\n";
    cudaGetLastError();

    rtstatus = cudaMemset(d_data, 0, 2*GB);
    std::cout << "cudaMemset full allocation: " << cudaGetErrorString(rtstatus) << "\n";
    cudaGetLastError();

    rtstatus = cudaMemcpy(d_data, h_data, GB, cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy to device chunk: " << cudaGetErrorString(rtstatus) << "\n";
    cudaGetLastError();
    nvtxRangePop();
    // rtstatus = cudaMemcpy(d_data, h_data, 2*GB, cudaMemcpyHostToDevice);
    // std::cout << "cudaMemcpy to full allocation: " << cudaGetErrorString(rtstatus) << "\n";
    // cudaGetLastError();

    std::cout << "---init c_data with d_data "  << std::endl;
    // test for current data
    cudaError_t err;
    if(checkflag)
        dataCheck(c_data, d_data, 2*GB, err, 0);

    nvtxRangePushA("unmap Range");
    std::cout << "---unmap first part of vmm"  << std::endl;
    // test for unmap
    CUresult result = cuMemUnmap(deviceptr, sizeof(char) *GB); // unmap first part
    if (result != CUDA_SUCCESS) {
        const char* errMsg;
        cuGetErrorString(result, &errMsg);
        std::cerr << "cuMemUnmap failed: " << errMsg << std::endl;
        // std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }
    nvtxRangePop();

    std::cout << "---update the first part of c_data to get if it update as the remap done"  << std::endl;
    

    if(checkflag){
        std::cout << "---d_data second part to c_data"  << std::endl;
        dataCheck(c_data, d_data, GB, err, sizeof(char) *GB);
        std::cout << "---update the first part of c_data"  << std::endl;
        dataUpdate(c_data, GB);
    }

        
    std::cout << "---remap with the same allocationHandle with the first part"  << std::endl;
    nvtxRangePushA("remap Range");
    // remap
    status = cuMemMap(deviceptr, GB, 0, allocationHandle, 0);
    if (status != CUDA_SUCCESS) {
        const char* errMsg;
        cuGetErrorString(result, &errMsg);
        std::cerr << "cuMemUnmap failed: " << errMsg << std::endl;
        // std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }

    // status = cuMemMap(deviceptr+GB, GB, 0, allocationHandle_2, 0);
    // if (status != CUDA_SUCCESS) {
    //     const char* errMsg;
    //     cuGetErrorString(result, &errMsg);
    //     std::cerr << "cuMemUnmap failed: " << errMsg << std::endl;
    //     // std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(result) << std::endl;
    //     return -1;
    // }

    status = cuMemSetAccess(deviceptr, GB, accessDescriptors.data(), 1);
    assert(status == CUDA_SUCCESS);
    // status = cuMemSetAccess(deviceptr + GB, GB, accessDescriptors.data(), 1);
    // assert(status == CUDA_SUCCESS);
    nvtxRangePop();

    
    if(checkflag){
        std::cout << "---check now the d_data is the same or not"  << std::endl;
        dataCheck(c_data, d_data, 2*GB, err, 0);
    }




    // cudaEvent_t eventA, eventB;
    // cudaEventCreate(&eventA);
    // cudaEventCreate(&eventB);
    // float elapsed;

    // cudaEventRecord(eventA);
    // cudaMemcpy(d_data, h_data, GB, cudaMemcpyHostToDevice);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemcpy to device chunk: " << elapsed << " ms" << "\n";

    // cudaEventRecord(eventA);
    // cudaMemcpy(d_data + GB, h_data, GB, cudaMemcpyHostToDevice);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemcpy to host chunk: " << elapsed << " ms" << "\n";

    // cudaEventRecord(eventA);
    // cudaMemcpy(d_data, h_data, 2*GB, cudaMemcpyHostToDevice);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemcpy to both: " << elapsed << " ms" << "\n";

    // cudaEventRecord(eventA);
    // cudaMemset(d_data, 0, GB);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemset device chunk: " << elapsed << " ms" << "\n";

    // cudaEventRecord(eventA);
    // cudaMemset(d_data + GB, 0, GB);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemset host chunk: " << elapsed << " ms" << "\n";

    // cudaEventRecord(eventA);
    // cudaMemset(d_data, 0, 2*GB);
    // cudaEventRecord(eventB);
    // cudaEventSynchronize(eventB);
    // cudaEventElapsedTime(&elapsed, eventA, eventB);
    // std::cout << "cudaMemset both: " << elapsed << " ms" << "\n";
    status = cuMemUnmap(deviceptr, allocationSize);
    assert(status == CUDA_SUCCESS);
    status = cuMemAddressFree(deviceptr, allocationSize);
    assert(status == CUDA_SUCCESS);
}