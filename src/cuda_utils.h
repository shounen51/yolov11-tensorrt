#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include "Logger.h"
#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr) \
    { \
        cudaError_t error_code = callstr; \
        if (error_code != cudaSuccess) { \
            AILOG_ERROR(std::string("CUDA error ") + std::to_string(error_code) + \
                " (" + cudaGetErrorString(error_code) + ")"); \
            assert(0); \
        } \
    }
#endif  // CUDA_CHECK

#endif  // TRTX_CUDA_UTILS_H_
