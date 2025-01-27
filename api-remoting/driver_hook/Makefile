CXX=g++
CUDAPATH?=/usr/local/cuda
NVCC ?= nvcc

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
all: libcudahook.so vector_add_driver vector_add_kernel.ptx
endif

COMMONFLAGS=-lcudart -Wall -fPIC -shared -ldl 

libcudahook.so: driver_hook.cpp
	$(CXX) -I$(CUDAPATH)/include -L$(CUDAPATH)/lib64 $(COMMONFLAGS) -o libcudahook.so driver_hook.cpp

vector_add_driver: vector_add_driver.cu
	$(NVCC) $^ -o $@ -lcuda -std=c++11

vector_add_kernel.ptx: vector_add_kernel.cu
	$(NVCC) -ptx -o $@  $^ -lcuda -std=c++11

clean:
	-rm libcudahook.so vector_add_driver vector_add_kernel.ptx