#pragma once

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "redis_ai_objects/script_struct.h"

void torchBasicTest();

DLManagedTensor *torchNewTensor(DLDataType dtype, long ndims, int64_t *shape, int64_t *strides,
                                char *data);

void *torchCompileScript(const char *script, DLDeviceType device, int64_t device_id, char **error,
                         void *(*alloc)(size_t));

void *torchLoadModel(const char *model, size_t modellen, DLDeviceType device, int64_t device_id,
                     char **error, void *(*alloc)(size_t));

bool torchMatchScriptSchema(size_t nArguments, long nInputs,
                            TorchScriptFunctionArgumentType *argumentTypes, size_t nlists,
                            size_t nOtherInputs, char **error);

void torchRunScript(void *scriptCtx, const char *fnName, long nInputs, DLManagedTensor **inputs,
                    long nOutputs, DLManagedTensor **outputs, size_t nArguments,
                    TorchScriptFunctionArgumentType *argumentTypes, size_t *listSizes,
                    RedisModuleString **otherInputs, char **error, void *(*alloc)(size_t));

void torchRunModel(void *modelCtx, long nInputs, DLManagedTensor **inputs, long nOutputs,
                   DLManagedTensor **outputs, char **error, void *(*alloc)(size_t));

void torchSerializeModel(void *modelCtx, char **buffer, size_t *len, char **error,
                         void *(*alloc)(size_t));

void torchDeallocContext(void *ctx);

void torchSetInterOpThreads(int num_threads, char **error, void *(*alloc)(size_t));

void torchSetIntraOpThreads(int num_threadsm, char **error, void *(*alloc)(size_t));

size_t torchModelNumInputs(void *modelCtx, char **error);

const char *torchModelInputNameAtIndex(void *modelCtx, size_t index, char **error);

size_t torchModelNumOutputs(void *modelCtx, char **error);

size_t torchScript_FunctionCount(void *scriptCtx);

const char *torchScript_FunctionName(void *scriptCtx, size_t fn_index);

size_t torchScript_FunctionArgumentCount(void *scriptCtx, size_t fn_index);

TorchScriptFunctionArgumentType torchScript_FunctionArgumentype(void *scriptCtx, size_t fn_index,
                                                                size_t arg_index);

#ifdef __cplusplus
}
#endif
