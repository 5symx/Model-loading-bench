#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <glog/logging.h>
#include <any>
#include <cuda.h>


// #include "../include/device_service_msgs.hpp"
// #include "device_memory.hpp"
// #include "device_function.hpp"

namespace service::compute {

class virtual_device;
class virtual_context;
class device_function;
class device_stream;
class device_kernel;
class device_memory;
//event
//graph
//module

class virtual_device { // CUdevice
public:

    fractos::core::future<void> Init_dev(fractos::wire::endian::uint8_t dev_id); // cuInit(0);
    
    fractos::core::future<virtual_device> get_device(
        fractos::wire::endian::uint8_t dev_id,
        const std::unordered_map<std::string, std::any>& backend_args); // id

    // fractos::core::future<device_memory> allocate_memory(mem_alloc_type type, size_t size, std::set<virtual_context> _dev_ctx); // type size ctx

    fractos::core::future<void> destroy();

    virtual_device(std::shared_ptr<void> pimpl);

    virtual_device(fractos::wire::endian::uint8_t id);

    ~virtual_device();

public:
    std::shared_ptr<virtual_device> _self;
    std::shared_ptr<void> _pimpl;
    fractos::wire::endian::uint8_t id;
    std::set<virtual_context> _dev_ctx;



private:
    bool _destroyed;
};

class virtual_context { //  // CUcontext 
public:

    fractos::core::future<virtual_context> create_context(std::weak_ptr<virtual_device> _dev);

    fractos::core::future<void> ctx_sync();

    fractos::core::future<void> ctx_destroy();

    virtual_context();

    ~virtual_context();

public:
    std::shared_ptr<virtual_context> _self;

    std::weak_ptr<virtual_device> _dev;


private:
    bool _destroyed;
};


class device_function { // CUfunction
public:
    fractos::core::future<device_function> get_function(std::weak_ptr<CUmodule> _module);

    fractos::core::future<void> unregister_func();

    device_function();

    ~device_function();

public:
    std::shared_ptr<device_function> _self;

    std::weak_ptr<CUmodule> _module;

private:
    
    bool _unregistered;
};


class device_stream { // CUfunction
public:
    fractos::core::future<device_stream> create_stream(std::weak_ptr<virtual_context> _ctx);

    fractos::core::future<void> destroy_kernel();

    device_stream();

    ~device_stream();

public:
    std::shared_ptr<device_stream> _self;

    std::weak_ptr<virtual_context> _ctx;

private:
    
    bool _destroyed;
};


class device_kernel { // CUfunction
public:
    fractos::core::future<device_kernel> launch_kernel(std::weak_ptr<device_stream> _stream,
                                                        const std::unordered_map<std::string, std::any>& backend_arg);

    fractos::core::future<void> unregister_kernel();

    device_kernel();

    ~device_kernel();

public:
    std::shared_ptr<device_kernel> _self;
    
    std::weak_ptr<device_stream> _stream;

private:
    
    bool _unregistered;
};


class device_memory { // CUfunction
public:
    fractos::core::future<CUdeviceptr> alloc_memory(size_t size, std::weak_ptr<virtual_context> _ctx, const std::unordered_map<std::string, std::any>& backend_arg); // dst ? 

    fractos::core::future<void> memcpyH2D( size_t size, std::weak_ptr<device_stream> _stream, 
                                            const std::unordered_map<std::string, std::any>& backend_arg); // dst, src, size
    
    fractos::core::future<void> memcpyD2H( size_t size, std::weak_ptr<device_stream> _stream, 
                                            const std::unordered_map<std::string, std::any>& backend_arg); // dst, src, size

    fractos::core::future<void> deallocate_memory();

    device_memory();

    ~device_memory();

public:
    std::shared_ptr<device_memory> _self;
    std::shared_ptr<CUdeviceptr> _devptr;

    std::weak_ptr<device_stream> _stream;
    std::weak_ptr<virtual_context> _ctx

private:
    bool _dealloc;
};

}
