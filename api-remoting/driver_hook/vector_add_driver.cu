#include <cuda.h>
#include <iostream>
#include <vector>

#define checkCudaErrors(err) \
    if (err != CUDA_SUCCESS) { \
        std::cerr << "CUDA error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Initialize the CUDA driver API
    checkCudaErrors(cuInit(0));

    // Get a handle to the first device
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));

    // Create a context
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // Load the PTX file
    CUmodule cuModule;
    checkCudaErrors(cuModuleLoad(&cuModule, "vector_add_kernel.ptx"));

    // Get a handle to the kernel function
    CUfunction vectorAdd;
    checkCudaErrors(cuModuleGetFunction(&vectorAdd, cuModule, "vectorAdd"));

    // Allocate device memory
    CUdeviceptr d_A, d_B, d_C;
    checkCudaErrors(cuMemAlloc(&d_A, size));
    checkCudaErrors(cuMemAlloc(&d_B, size));
    checkCudaErrors(cuMemAlloc(&d_C, size));

    // Copy data from host to device
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A.data(), size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B.data(), size));

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Set up kernel parameters
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // Launch the kernel
    checkCudaErrors(cuLaunchKernel(vectorAdd,
                                   blocksPerGrid, 1, 1,   // Grid dimensions
                                   threadsPerBlock, 1, 1, // Block dimensions
                                   0, nullptr,            // Shared memory and stream
                                   args, nullptr));       // Kernel parameters

    // Wait for the kernel to finish
    checkCudaErrors(cuCtxSynchronize());

    // Copy result from device to host
    checkCudaErrors(cuMemcpyDtoH(h_C.data(), d_C, size));

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    // Free device memory
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));

    // Destroy the context
    checkCudaErrors(cuCtxDestroy(cuContext));

    return 0;
}