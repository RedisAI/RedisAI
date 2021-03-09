#pragma once

#include "limits.h"
#include "config/config.h"
#include "dlpack/dlpack.h"

#define LEN_UNKOWN ULONG_MAX
typedef struct RAI_Tensor {
    DLManagedTensor tensor;
    size_t len;
    long long refCount;
} RAI_Tensor;
