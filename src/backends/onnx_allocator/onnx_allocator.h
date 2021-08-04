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
