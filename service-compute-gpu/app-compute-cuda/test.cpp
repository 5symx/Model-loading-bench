#include <stdio.h>
#include <cmath>


#include <iostream>
#include <string.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>


#define checkCudaErrors(err)  handleError(err, __FILE__, __LINE__)

void handleError(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        std::cout << "CUDA Driver API error = " << err
                    << " from file <" << file << ">, line " << line << ".\n";
        exit(-1);
    }
    std::cout << "CUDA Driver API SUCCESS from file <" << file << ">, line " << line << ".\n";
}

void profile_exp(size_t N)
{
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_A, d_B, d_C;
    size_t size = N * sizeof(int);
    checkCudaErrors(cuCtxCreate(&context, 0, 0));

    // CUstream stream;
    // checkCudaErrors(cuStreamCreate(&stream, 0));
    // checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    // unsigned int flag = (int)CU_STREAM_DEFAULT;
    // std::cout << flag << std::endl;


    nvtxRangeId_t rangeId_A = nvtxRangeStartA("memalloc");
    checkCudaErrors(cuMemAlloc(&d_A, size));
    checkCudaErrors(cuMemAlloc(&d_B, size));
    checkCudaErrors(cuMemAlloc(&d_C, size));
    nvtxRangeEnd(rangeId_A);

    nvtxRangeId_t rangeId_H = nvtxRangeStartA("host_alloc");
    // Initialize host arrays

    // int *pinnedMemory;
    // int* h_A = (int*)malloc(size);
    // int* h_B = (int*)malloc(size);
    // int* h_C = (int*)malloc(size);
    
    int* h_A;
    int* h_B;
    int* h_C;

    checkCudaErrors(cuMemHostAlloc((void**)&h_A, size, CU_MEMHOSTALLOC_PORTABLE));
    checkCudaErrors(cuMemHostAlloc((void**)&h_B, size, CU_MEMHOSTALLOC_PORTABLE));
    checkCudaErrors(cuMemHostAlloc((void**)&h_C, size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);

        // h_C[i] = static_cast<int>(0);
    }
    
    nvtxRangeEnd(rangeId_H);

    nvtxRangeId_t rangeId_B = nvtxRangeStartA("memcpyH2D");
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));
    nvtxRangeEnd(rangeId_B);


    std::string ptxSource = "test.ptx";
    nvtxRangeId_t rangeId_C = nvtxRangeStartA("load_kernel");
    checkCudaErrors(cuModuleLoad(&module, ptxSource.c_str()));
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "add"));
    nvtxRangeEnd(rangeId_C);

    nvtxRangeId_t rangeId_D = nvtxRangeStartA("launch_kernel + sync");
    void* args[] = { &d_A, &d_B, &d_C, &N };
    checkCudaErrors(cuLaunchKernel(kernel, 1024, 1, 1, 1024, 1, 1, 0, 0, args, 0));
    checkCudaErrors(cuCtxSynchronize());
    nvtxRangeEnd(rangeId_D);

    nvtxRangeId_t rangeId_E = nvtxRangeStartA("memcpyD2H");
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));
    nvtxRangeEnd(rangeId_E);

    // // Stop profiling
    // cudaProfilerStop();

    // Verify the result
    bool resultIsCorrect = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            resultIsCorrect = false;
            std::cout << i << "  " << N << std::endl;
            break;
        }
    }
    

    if (resultIsCorrect) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }
    cuMemFreeHost((void*)h_A);
    cuMemFreeHost((void*)h_B);
    cuMemFreeHost((void*)h_C);

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

}

void profile_copy(size_t N)
{
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_A, d_B, d_C;
    size_t size = N * sizeof(int);
    checkCudaErrors(cuCtxCreate(&context, 0, 0));

    // CUstream stream;
    // checkCudaErrors(cuStreamCreate(&stream, 0));
    // checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    // unsigned int flag = (int)CU_STREAM_DEFAULT;
    // std::cout << flag << std::endl;


    nvtxRangeId_t rangeId_A = nvtxRangeStartA("memalloc");
    checkCudaErrors(cuMemAlloc(&d_A, size));
    // checkCudaErrors(cuMemAlloc(&d_B, size));
    // checkCudaErrors(cuMemAlloc(&d_C, size));
    nvtxRangeEnd(rangeId_A);

    nvtxRangeId_t rangeId_H = nvtxRangeStartA("host_alloc");



    
    int* h_A;
    // int* h_B;
    int* h_C;

    checkCudaErrors(cuMemHostAlloc((void**)&h_A, size, CU_MEMHOSTALLOC_PORTABLE));
    // checkCudaErrors(cuMemHostAlloc((void**)&h_B, size, CU_MEMHOSTALLOC_PORTABLE));
    checkCudaErrors(cuMemHostAlloc((void**)&h_C, size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        // h_B[i] = static_cast<int>(i * 2);

        // h_C[i] = static_cast<int>(0);
    }
    
    nvtxRangeEnd(rangeId_H);

    nvtxRangeId_t rangeId_B = nvtxRangeStartA("memcpyH2D");
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    // checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));
    nvtxRangeEnd(rangeId_B);


    nvtxRangeId_t rangeId_E = nvtxRangeStartA("memcpyD2H");
    checkCudaErrors(cuMemcpyDtoH(h_C, d_A, size));
    nvtxRangeEnd(rangeId_E);

    // // Stop profiling
    // cudaProfilerStop();

    // Verify the result
    bool resultIsCorrect = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i])) > 1e-5) {
            resultIsCorrect = false;
            std::cout << i << "  " << N << std::endl;
            break;
        }
    }
    

    if (resultIsCorrect) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }
    cuMemFreeHost((void*)h_A);
    // cuMemFreeHost((void*)h_B);
    cuMemFreeHost((void*)h_C);

    cuMemFree(d_A);
    // cuMemFree(d_B);
    // cuMemFree(d_C);
    // cuModuleUnload(module);
    cuCtxDestroy(context);

}

int main() {
    
    size_t N = 1024*1024;

    checkCudaErrors(cuInit(0));

    for(int i = 0; i < 30; i++)
    {
        // profile_copy(N);
        profile_exp(N);
    }
    
    return 0;
}