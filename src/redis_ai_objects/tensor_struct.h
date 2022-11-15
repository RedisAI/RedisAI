/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "limits.h"
#include "config/config.h"
#include "dlpack/dlpack.h"

#define LEN_UNKNOWN ULONG_MAX
typedef struct RAI_Tensor {
    DLManagedTensor tensor;
    size_t len;
    long long refCount;
    size_t blobSize;
} RAI_Tensor;
