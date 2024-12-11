# GPU direct benchmark

## Update Makefile for benchmark
1. update Makefile and `make`
2. test with samples `./cufile_sample_001 file_path 0`

## FIO test
1. [fio](https://github.com/axboe/fio) install with cuda `./configure --enable-cuda --enable-libcufile && make && make install`
2. Test cufile `./fio examples/libcufile-cufile.fio`

## Trouble shooting
1. cuda setting in make `CFLAGS = -I$(CUDA_PATH)/include ` `LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda`
2. cuda head file `#include<cuda.h>` `#include<cuda_runtime.h>`
# REFERENCE

[INSTALL](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#troubleshoot-install)

[API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)

[CODE-EXAMPLE](https://github.com/NVIDIA/MagnumIO/tree/main/gds/samples)
