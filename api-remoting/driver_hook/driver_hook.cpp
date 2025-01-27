#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
//#include <cuda_runtime.h>
//#include <driver_types.h>

void cuLaunchKernelHelper (CUstream hStream);


CUresult cuLaunchKernel (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {

        void* handle;
        CUresult (*function)(CUfunction f,  
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

        *(void **)(&function) = dlsym (RTLD_NEXT, "cuLaunchKernel");

        cuLaunchKernelHelper (hStream);

        (*function)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

}

void cuLaunchKernelHelper (CUstream hStream) {
        // Nothing
        printf ("cuLaunchHelper\n");
}