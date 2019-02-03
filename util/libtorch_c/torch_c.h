#ifndef __TORCH_C_H__
#define __TORCH_C_H__

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

void torchBasicTest();

DLManagedTensor* torchNewTensor(DLDataType dtype, long ndims,
                                long long* shape, long long* strides,
                                char* data);

void* torchCompileScript(const char* script, DLDeviceType device);

void* torchLoadGraph(const char* graph, DLDeviceType device);

long torchRunScript(void* scriptCtx, const char* fnName,
                    long nInputs, DLManagedTensor** inputs,
                    long nOutputs, DLManagedTensor** outputs);

long torchRunGraph(void* graphCtx,
                   long nInputs, DLManagedTensor** inputs,
                   long nOutputs, DLManagedTensor** outputs);

void torchDeallocContext(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // __TORCH_C_H__
