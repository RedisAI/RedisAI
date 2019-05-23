#ifndef __TORCH_C_H__
#define __TORCH_C_H__

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

void torchBasicTest();

DLManagedTensor* torchNewTensor(DLDataType dtype, long ndims,
                                int64_t* shape, int64_t* strides,
                                char* data);

void* torchCompileScript(const char* script, DLDeviceType device,
                         char **error, void* (*alloc)(size_t));

void* torchLoadModel(const char* model, size_t modellen, DLDeviceType device,
                     char **error, void* (*alloc)(size_t));

void torchRunScript(void* scriptCtx, const char* fnName,
                    long nInputs, DLManagedTensor** inputs,
                    long nOutputs, DLManagedTensor** outputs,
                    char **error, void* (*alloc)(size_t));

void torchRunModel(void* modelCtx,
                   long nInputs, DLManagedTensor** inputs,
                   long nOutputs, DLManagedTensor** outputs,
                   char **error, void* (*alloc)(size_t));

void torchSerializeModel(void* modelCtx, char **buffer, size_t *len,
                         char **error, void* (*alloc)(size_t));

void torchDeallocContext(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // __TORCH_C_H__
