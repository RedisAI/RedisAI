#ifndef __TORCH_C_H__
#define __TORCH_C_H__

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

void torchBasicTest();

void* torchCompileScript(const char* script);

void torchDeallocScript(void* scriptCtx);

#ifdef __cplusplus
}
#endif

#endif // __TORCH_C_H__
