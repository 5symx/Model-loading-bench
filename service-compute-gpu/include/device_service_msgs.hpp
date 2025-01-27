/*
**    Definitions for communicaiton messages for device_service
*/

#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/wire/endian.hpp>

namespace service::compute{
  
    namespace detail::device_service {
        struct make_virtual_device {
            struct request {
                struct imms {
                    fractos::wire::endian::uint8_t id;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request allocate_memory;
                    fractos::core::cap::request register_function;
                    fractos::core::cap::request destroy;
                };
            };
        };

        struct get_virtual_device {
            struct request {
                struct imms {
                    fractos::wire::endian::uint8_t id;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request allocate_memory;
                    fractos::core::cap::request register_function;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    namespace detail::virtual_device {
        struct allocate_memory {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t virtual_device_id;
                    fractos::wire::endian::uint64_t type;
                    fractos::wire::endian::uint64_t size;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t address;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::memory memory;
                    fractos::core::cap::request deallocate;
                };
            };
        };

        struct register_function {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t virtual_device_id;
                    fractos::wire::endian::uint8_t func_id;
                    fractos::wire::endian::uint64_t func_name_size;
                    char func_name[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::memory cuda_file;
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request call;
                    fractos::core::cap::request unregister;
                };
            };
        };

        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }
    
    namespace detail::device_function {
        struct call {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t args_num;
                    fractos::wire::endian::uint64_t grid;
                    fractos::wire::endian::uint64_t block;
                    char kernel_args[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation_success; 
                    fractos::core::cap::request continuation_failure;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
            struct kernel_arg_info {
                fractos::wire::endian::uint64_t size;
                char value[];
            };
        };

        struct unregister {
            struct request {
                struct imms {
                    fractos::wire::endian::uint8_t func_id;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }
    
    namespace detail::device_memory {
        struct deallocate {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }
}

