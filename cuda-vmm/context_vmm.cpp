#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <cuda.h>
#include <cstring>
typedef int ShareableHandle;
#define DATA_BUF_SIZE 4ULL * 1024ULL * 1024ULL


#define checkCudaErrors(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__) // from https://liwuhen.cn/model_deploy/cuda_driverapi/

#define CHECK_DRV(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__) // from https://liwuhen.cn/model_deploy/cuda_driverapi/

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){
    if(code != CUresult::CUDA_SUCCESS){    
        const char* err_name = nullptr;    
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int memKernelFunction(CUdeviceptr d_ptr){
    
    CUcontext ctx2;
    CUdevice device;
    CUstream stream;
    int devIdx;
    devIdx = 0;
    // SPRINTF(devIdx, "%d", selectedDevices[0]);
    CHECK_DRV(cuDeviceGet(&device, devIdx));
    CHECK_DRV(cuCtxCreate(&ctx2, 0, device));
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    
    CUmodule module;
    CUfunction function;

    // Load the module (assuming module is already loaded)
    checkCudaErrors(cuModuleLoad(&module, "memMapIpc_kernel.ptx"));

    // Get the function handle
    checkCudaErrors(cuModuleGetFunction(&function, module, "memMapIpc_kernel"));

     // Build arguments to be passed to cuda kernel.
    CUdeviceptr ptr_tmp = d_ptr;
    int size = DATA_BUF_SIZE;
    char val = (char)1;

    void *args[] = {&ptr_tmp, &size, &val};
    int blocks = 0;
    int threads = 128;

    int multiProcessorCount;
    checkCudaErrors(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, function, threads, 0));
    checkCudaErrors(cuDeviceGetAttribute(
        &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    blocks *= multiProcessorCount;


    // Push a simple kernel on th buffer.
    checkCudaErrors(cuLaunchKernel(function, blocks, 1, 1, threads, 1,
                                   1, 0, stream, args, 0));
    checkCudaErrors(cuStreamSynchronize(stream));

}
int memMapValidation(CUdeviceptr d_ptr){

    CUstream stream;
    int devIdx;
    devIdx = 0;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    printf("Process %d: verifying...\n", devIdx);

    std::vector<char> verification_buffer(DATA_BUF_SIZE);
    checkCudaErrors(cuMemcpyDtoHAsync(&verification_buffer[0],
                                        d_ptr , DATA_BUF_SIZE,
                                        stream));
    checkCudaErrors(cuStreamSynchronize(stream));

    // The contents should have the id of the sibling just after me
    char compareId = (char)1; // same as var in function
    for (unsigned long long j = 0; j < DATA_BUF_SIZE; j++) {
        if (verification_buffer[j] != compareId) {
        printf("Process %d: Verification mismatch at %lld: %d != %d\n", devIdx, j,
                (int)verification_buffer[j], (int)compareId);
        break;
        }
    }
}


int memMapAllocateAndExportMemory(int id, ShareableHandle &shdl, CUmemAllocationHandleType ipcHandleTypeFlag) {

    // ShareableHandle shdl; // handle reuse through ipc 
    CUmemGenericAllocationHandle hdl;
    
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = ipcHandleTypeFlag; // need for share hdl

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.location.id = 0;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUdeviceptr ptr;

    size_t sz = DATA_BUF_SIZE;
    // size_t desired_size = 512 * 210 * 210 * 2 * sizeof(float);
    size_t aligned_sz; 

    if (cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
        std::cerr << "Failed to get granularity allocation" << std::endl;
        return 1;
    }
    // sz = ((desired_size + aligned_sz - 1) / aligned_sz) * aligned_sz;

    if (sz % aligned_sz) {
        printf(
            "Allocation size is not a multiple of minimum supported granularity "
            "for this device. Exiting...\n");
        exit(EXIT_FAILURE);
  }

    CHECK_DRV(cuMemAddressReserve(&ptr, DATA_BUF_SIZE, DATA_BUF_SIZE, 0, 0)); // VA address
    CHECK_DRV(cuMemCreate(&hdl, DATA_BUF_SIZE, &prop, 0)); // physical memory from device
    std::cout << "ptr: " << ptr << std::endl;
    CHECK_DRV(cuMemMap(ptr, DATA_BUF_SIZE, 0, hdl, 0));
    checkCudaErrors(cuMemSetAccess(ptr, DATA_BUF_SIZE, &accessDescriptor, 1));


    
    memKernelFunction(ptr);
    memMapValidation(ptr);

    CHECK_DRV( cuMemExportToShareableHandle(&shdl, hdl, ipcHandleTypeFlag, 0)); // for this physical memory may be need another process 
    std::cout << hdl << ", " << shdl << std::endl;
    // return shdl;
}
int memMapImportAndMapMemory( ShareableHandle shdl_init, CUmemAllocationHandleType ipcHandleTypeFlag) {
    ShareableHandle shdl = shdl_init;
    // // update to another ctx and VA address

    CUmemGenericAllocationHandle hdl2; // new for it 
    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.location.id = 0;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    CUdeviceptr d_ptr = 0ULL;

    // Reserve the required contiguous VA space for the allocations
    CHECK_DRV(cuMemAddressReserve(&d_ptr, DATA_BUF_SIZE,
                                        DATA_BUF_SIZE, 0, 0));


    
    CHECK_DRV(cuMemImportFromShareableHandle(&hdl2, (void *)(uintptr_t)shdl, ipcHandleTypeFlag));


    std::cout << hdl2 << ", " << shdl << std::endl;

    CHECK_DRV(cuMemMap(d_ptr, DATA_BUF_SIZE, 0ULL, hdl2, 0ULL));
    checkCudaErrors(cuMemSetAccess(d_ptr, DATA_BUF_SIZE, &accessDescriptor, 1));

    std::cout << "d_ptr: "  << d_ptr << std::endl;
    memMapValidation(d_ptr);
}

int main(int argc, char **argv) {
    cuInit(0);

    std::vector<std::thread> threads;
    // std::vector<ShareableHandle> shdl(1);
    ShareableHandle shdl;
    
    CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; //  CU_MEM_HANDLE_TYPE_WIN32
    // test();

    for (int i = 0; i < 1; ++i) {
        memMapAllocateAndExportMemory(i, shdl, ipcHandleTypeFlag);
        std::cout <<"check shdl " << shdl << std::endl;
        memMapImportAndMapMemory(shdl, ipcHandleTypeFlag);
        // threads.emplace_back(memMapAllocateAndExportMemory, i, &shdl, ipcHandleTypeFlag); // initial new workerThread in place 
        // time.sleep(1);
        // threads.emplace_back(memMapImportAndMapMemory, &shdl, ipcHandleTypeFlag); // initial new workerThread in place 
    }

    

    for (auto& thread : threads) {
        thread.join();
    }
    return 0;
}