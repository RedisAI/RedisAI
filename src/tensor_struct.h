#pragma once

#include "config.h"
#include "../deps/linux-x64-cpu/dlpack/include/dlpack/dlpack.h"
//#include "dlpack/dlpack.h"
#include "limits.h"

#define LEN_UNKOWN ULONG_MAX
typedef struct RAI_Tensor {
    DLManagedTensor tensor;
    size_t len;
    long long refCount;
} RAI_Tensor;
