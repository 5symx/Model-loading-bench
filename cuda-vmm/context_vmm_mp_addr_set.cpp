#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <cuda.h>
// #include <string.h>
#include <cstring>
#include "helper_multiprocess.h"

typedef int ShareableHandle;

static const char ipcName[] = "memmap_ipc_pipe";
static const char shmName[] = "memmap_ipc_shm";

#define DATA_BUF_ADDRESS (CUdeviceptr)0x7f3000000000
#define DATA_BUF_ADDRESS (CUdeviceptr)0x7fc000000000
#define DATA_BUF_ADDRESS (CUdeviceptr)0x7fd000000000

#define DATA_BUF_RESERVE_SIZE (4ULL << 30)
#define DATA_BUF_SIZE (2ULL << 30)
#define DATA_BUF_ALIGN (1ULL << 30)
#define VAR_ID 42

#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)

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


typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
  CUdeviceptr ptr;
} shmStruct;


static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n) {
  int count;

  // Check-in
  count = cpu_atomic_add32(barrier, 1);
  if (count == n) {  // Last one in
    *sense = 1;
  }
  while (!*sense)
    ;

  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  if (count == 0) {  // Last one out
    *sense = 0;
  }
  while (*sense)
    ;
}




void setMemory(CUdeviceptr d_ptr){
    
    int id = 0;
    // update to another ctx and VA address

    CUcontext ctx2;
    CUdevice device;
    CUstream stream;
    int devIdx;
    devIdx = 0;
    // SPRINTF(devIdx, "%d", selectedDevices[0]);
    printf("Process %d: setting ....\n", devIdx);


    CHECK_DRV(cuDeviceGet(&device, devIdx));
    CHECK_DRV(cuCtxCreate(&ctx2, 0, device));
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CUmodule module;
    CUfunction function;
    CUresult result;

    // Load the module (assuming module is already loaded)
    checkCudaErrors(cuModuleLoad(&module, "memMapIpc_kernel.ptx"));

    // Get the function handle
    checkCudaErrors(cuModuleGetFunction(&function, module, "memMapIpc_kernel"));
    
    uintptr_t size = DATA_BUF_SIZE;
    char val = (char)VAR_ID;

    void *args[] = {&d_ptr, &size, &val};
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

    printf("Process %d: setting done.\n", devIdx);
}

void verifyMemory(CUdeviceptr d_ptr){

    CUcontext ctx2;
    CUdevice device;
    CUstream stream;
    int devIdx;
    devIdx = 0;
    // SPRINTF(devIdx, "%d", selectedDevices[0]);


    CHECK_DRV(cuDeviceGet(&device, devIdx));
    CHECK_DRV(cuCtxCreate(&ctx2, 0, device));
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    int id = 0;
  printf("Process %d: verifying...\n", id);

  std::vector<char> verification_buffer(DATA_BUF_SIZE);
  checkCudaErrors(cuMemcpyDtoHAsync(&verification_buffer[0],
                                    d_ptr , DATA_BUF_SIZE,
                                    stream));
  checkCudaErrors(cuStreamSynchronize(stream));

  // The contents should have the id of the sibling just after me
  char compareId = (char)VAR_ID;
  for (unsigned long long j = 0; j < DATA_BUF_SIZE; j++) {
    if (verification_buffer[j] != compareId) {
      printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
             (int)verification_buffer[j], (int)compareId);
      break;
    }
  }
}



CUdeviceptr memMapImportAndMapMemory( ShareableHandle *shdl, CUmemAllocationHandleType ipcHandleTypeFlag, CUdeviceptr ptr_tras) {

    // std::cout << "second: " << shdl <<" "<<  *shdl << std::endl;

    CUmemGenericAllocationHandle hdl;
    // size_t sz;
    // size_t desired_size = 512 * 210 * 210 * 2 * sizeof(float);

  // The accessDescriptor will describe the mapping requirement for the
  // mapDevice passed as argument
    CUdeviceptr ptr;
    size_t aligned_sz;
    CUmemAccessDesc accessDesc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = ipcHandleTypeFlag;

    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUcontext ctx;
    
    // CHECK_DRV(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, 0));

    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    checkCudaErrors(cuCtxCreate(&ctx, 0, device));
    CHECK_DRV(cuCtxSetCurrent(ctx));

    std::cout << "import ptr: " << std::hex << ptr_tras << std::endl;
    CHECK_DRV(cuMemAddressReserve(&ptr, DATA_BUF_RESERVE_SIZE, DATA_BUF_ALIGN, ptr_tras, 0)); // VA address
    // checkCudaErrors(cuMemAddressFree(ptr, DATA_BUF_SIZE));

    // std::cout << "new actual ptr: " << ptr << std::endl;
    // CHECK_DRV(cuMemAddressReserve(&ptr, DATA_BUF_SIZE, DATA_BUF_SIZE, ptr, 0)); // VA address
    std::cout << "new actual ptr: " << std::hex << ptr << std::endl;

    // Import the memory allocation back into a CUDA handle from the platform
    // specific handle.
    CHECK_DRV(cuMemImportFromShareableHandle(
        &hdl,  (void *)(uintptr_t)*shdl,
        ipcHandleTypeFlag));

   
    CHECK_DRV(cuMemMap(ptr, DATA_BUF_SIZE, 0ULL, hdl, 0ULL));
    

    CHECK_DRV(cuMemSetAccess(ptr, DATA_BUF_SIZE, &accessDesc, 1ULL)); // can be use as malloc after this.
       

    verifyMemory(ptr);

    // CHECK_DRV(cuMemUnmap(ptr, DATA_BUF_SIZE));
    // CHECK_DRV(cuMemAddressFree(ptr, DATA_BUF_SIZE));
    CHECK_DRV(cuMemRelease(hdl));

    return ptr;

}


CUdeviceptr memMapAllocateAndExportMemory(ShareableHandle *shdl, CUmemAllocationHandleType ipcHandleTypeFlag) {


    int id = 0 ;

    // ShareableHandle shdl;// handle reuse through ipc 
    CUmemGenericAllocationHandle hdl;
    

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = ipcHandleTypeFlag; // need for share hdl

    CUmemAccessDesc accessDesc;
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // CUcontext ctx;
    // CHECK_DRV(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, 0));
    // CHECK_DRV(cuCtxSetCurrent(ctx));
    CUdeviceptr ptr;

    size_t sz;
    size_t desired_size = 512 * 210 * 210 * 2 * sizeof(float);
    size_t aligned_sz; 

    if (cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
        std::cerr << "Failed to get granularity allocation" << std::endl;
        return 1;
    }
    sz = ((desired_size + aligned_sz - 1) / aligned_sz) * aligned_sz;

    if (sz % aligned_sz) {
        printf(
            "Allocation size is not a multiple of minimum supported granularity "
            "for this device. Exiting...\n");
        exit(EXIT_FAILURE);
  }
    // std::cout << "import ptr: " << ptr << std::endl;
    if (false) {
      printf("pid= %d\n", getpid());
      char buf;
      read(0, &buf, 1);
    }
    CHECK_DRV(cuMemAddressReserve(&ptr, DATA_BUF_RESERVE_SIZE, DATA_BUF_ALIGN, DATA_BUF_ADDRESS, 0)); // VA address
    if (ptr != DATA_BUF_ADDRESS)
    {
      exit(EXIT_FAILURE);

    }
    // checkCudaErrors(cuMemAddressFree(ptr, DATA_BUF_SIZE));

    // std::cout << "new import ptr: " << ptr << std::endl;
    // CHECK_DRV(cuMemAddressReserve(&ptr, DATA_BUF_SIZE, DATA_BUF_SIZE, ptr, 0)); // VA address
    std::cout << "new import ptr: " << std::hex << ptr << std::endl;

    CHECK_DRV(cuMemCreate(&hdl, DATA_BUF_SIZE, &prop, 0)); // physical memory from device

    CHECK_DRV(cuMemMap(ptr, DATA_BUF_SIZE, 0ULL, hdl, 0ULL));

    CHECK_DRV(cuMemSetAccess(ptr, DATA_BUF_SIZE, &accessDesc, 1ULL));
    
    CHECK_DRV( cuMemExportToShareableHandle(shdl, hdl, ipcHandleTypeFlag, 0)); // for this physical memory may be need another process 
    // std::cout << "in-place: "<< hdl << ", " << *shdl << std::endl;

    setMemory(ptr);


    return ptr;

}

void memMapUnmapAndFreeMemory(CUdeviceptr dptr, size_t size) {
  CUresult status = CUDA_SUCCESS;

  // Unmap the mapped virtual memory region
  // Since the handles to the mapped backing stores have already been released
  // by cuMemRelease, and these are the only/last mappings referencing them,
  // The backing stores will be freed.
  // Since the memory has been unmapped after this call, accessing the specified
  // va range will result in a fault (unitll it is remapped).
  checkCudaErrors(cuMemUnmap(dptr, size));

  // Free the virtual address region.  This allows the virtual address region
  // to be reused by future cuMemAddressReserve calls.  This also allows the
  // virtual address region to be used by other allocation made through
  // opperating system calls like malloc & mmap.
  checkCudaErrors(cuMemAddressFree(dptr, DATA_BUF_RESERVE_SIZE));
}


static void childProcess(int devId, int id, char **argv) {
    devId = 0;

    CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; //  CU_MEM_HANDLE_TYPE_WIN32

    CUcontext ctx;
    CUdevice device;
    CUstream stream;
    int multiProcessorCount;

    checkCudaErrors(cuDeviceGet(&device, devId));
    checkCudaErrors(cuCtxCreate(&ctx, 0, device));
    checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    volatile shmStruct *shm = NULL;
    sharedMemoryInfo info;
    ipcHandle *ipcChildHandle = NULL;
    int blocks = 0;
    int threads = 128;

    checkIpcErrors(ipcOpenSocket(ipcChildHandle));

    if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    int procCount = (int)shm->nprocesses;

    barrierWait(&shm->barrier, &shm->sense, (unsigned int)(procCount + 1));
    // printf(" get out of child process sync.\n");

    ShareableHandle shdl;

    checkIpcErrors(ipcRecvShareableHandle(ipcChildHandle, &shdl));

    CUdeviceptr d_ptr = memMapImportAndMapMemory(&shdl, ipcHandleTypeFlag, shm->ptr);

    memMapUnmapAndFreeMemory(d_ptr, DATA_BUF_RESERVE_SIZE);
    checkIpcErrors(ipcCloseShareableHandle(shdl));
    checkIpcErrors(ipcCloseSocket(ipcChildHandle));
    checkCudaErrors(cuStreamDestroy(stream));
    checkCudaErrors(cuCtxDestroy(ctx));



}

static void parentProcess(char *app) {

  int devCount, i, nprocesses = 0;
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  std::vector<Process> processes;

  if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
  }

  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));
  nprocesses = 1;
  shm->nprocesses = 1;

// Initialize
  checkCudaErrors(cuInit(0));
  ShareableHandle shdl;
  // std::cout << "before: "<< shdl <<" "<<  &shdl << std::endl;
  CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; //  CU_MEM_HANDLE_TYPE_WIN32
  CUdeviceptr ptr = memMapAllocateAndExportMemory(&shdl, ipcHandleTypeFlag);
  // std::cout << "after: "<< shdl <<" "<<  &shdl << std::endl;
  shm->ptr = ptr;
  std::cout << "trans ptr  : "<< std::hex << ptr << std::endl;

  // Launch the child processes!

  char devIdx[10];
  char procIdx[10];
  char *const args[] = {app, devIdx, procIdx, NULL};
  Process process;

  sprintf(devIdx, "%d", 1);
  sprintf(procIdx, "%d", 1);

  if (spawnProcess(&process, app, args)) {
      printf("Failed to create process\n");
      exit(EXIT_FAILURE);
  }
  processes.push_back(process);

  barrierWait(&shm->barrier, &shm->sense, (unsigned int)(nprocesses + 1));
  // printf(" get out of parent process sync.\n");

  ipcHandle *ipcParentHandle = NULL;
  checkIpcErrors(ipcCreateSocket(ipcParentHandle, ipcName, processes));
  checkIpcErrors(
      ipcSendShareableHandle(ipcParentHandle, shdl, process));

  // Close the shareable handles as they are not needed anymore.
  for (int i = 0; i < nprocesses; i++) {
      checkIpcErrors(ipcCloseShareableHandle(shdl));
  }

  // And wait for them to finish

  if (waitProcess(&process) != EXIT_SUCCESS) {
  printf("Process %d failed!\n", i);
  exit(EXIT_FAILURE);
  }

  memMapUnmapAndFreeMemory(ptr, DATA_BUF_SIZE);

  checkIpcErrors(ipcCloseSocket(ipcParentHandle));
  sharedMemoryClose(&info);

}

int main(int argc, char **argv) {
    // Initialize
    checkCudaErrors(cuInit(0));

  if (argc == 1) {
    parentProcess(argv[0]);
  } else {
    childProcess(atoi(argv[1]), atoi(argv[2]), argv);
  }
  return EXIT_SUCCESS;
}
