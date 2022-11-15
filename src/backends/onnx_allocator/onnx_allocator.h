/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

OrtAllocator *CreateCustomAllocator(unsigned long long max_memory);

unsigned long long RAI_GetMemoryInfoORT();

unsigned long long RAI_GetMemoryAccessORT();

#ifdef __cplusplus
}
#endif
