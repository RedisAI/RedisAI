#pragma once

#include <stdint.h>
#include "redismodule.h"

#ifdef BACKENDS_API_EXTERN
#define BACKENDS_API extern
#endif

#ifndef BACKENDS_API
#define BACKENDS_API
#endif

typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Model RAI_Model;
typedef struct RAI_ModelRunCtx RAI_ModelRunCtx;
typedef struct RAI_Error RAI_Error;

/**
 * @return The internal id of RedisAI current working thread.
 * id range is {0, ..., <threads_count>-1}. If this is called from a non
 * RedisAI BG thread, return -1.
 */
BACKENDS_API long (*RedisAI_GetThreadId)(void);

/**
 * @return The number of working threads in RedisAI. This number should be
 * equal to the number of threads per queue (load time config) * number of devices
 * registered in RedisAI (a new device is registered if a model is set to run on
 * this device in AI.MODELSTORE command.
 */
BACKENDS_API uintptr_t (*RedisAI_GetThreadsCount)(void);

/**
 * @return The number of working threads per device queue (load time config).
 */
BACKENDS_API long long (*RedisAI_GetNumThreadsPerQueue)(void);

/**
 * @return The maximal number of milliseconds that a model run session should run
 * before it is terminated forcefully (load time config).
 * Currently supported only fo onnxruntime backend.
 */
BACKENDS_API long long (*RedisAI_GetModelExecutionTimeout)(void);

/**
 * The following functions are part of RedisAI low level API (the full low level
 * API is defined in redisai.h). For every function below named "RedisAI_X", its
 * implementation can be found under the name "RAI_X" in RedisAI header files.
 */

BACKENDS_API int (*RedisAI_InitError)(RAI_Error **err);
BACKENDS_API void (*RedisAI_FreeError)(RAI_Error *err);
BACKENDS_API const char *(*RedisAI_GetError)(RAI_Error *err);

BACKENDS_API RAI_Tensor *(*RedisAI_TensorCreateFromDLTensor)(DLManagedTensor *dl_tensor);
BACKENDS_API DLTensor *(*RedisAI_TensorGetDLTensor)(RAI_Tensor *tensor);
BACKENDS_API RAI_Tensor *(*RedisAI_TensorGetShallowCopy)(RAI_Tensor *t);
BACKENDS_API void (*RedisAI_TensorFree)(RAI_Tensor *tensor);

BACKENDS_API RAI_ModelRunCtx *(*RedisAI_ModelRunCtxCreate)(RAI_Model *model);
BACKENDS_API int (*RedisAI_GetModelFromKeyspace)(RedisModuleCtx *ctx, RedisModuleString *keyName,
                                                 RAI_Model **model, int mode, RAI_Error *err);
BACKENDS_API int (*RedisAI_ModelRunCtxAddInput)(RAI_ModelRunCtx *mctx, const char *inputName,
                                                RAI_Tensor *inputTensor);
BACKENDS_API int (*RedisAI_ModelRunCtxAddOutput)(RAI_ModelRunCtx *mctx, const char *outputName);
BACKENDS_API size_t (*RedisAI_ModelRunCtxNumOutputs)(RAI_ModelRunCtx *mctx);
BACKENDS_API RAI_Tensor *(*RedisAI_ModelRunCtxOutputTensor)(RAI_ModelRunCtx *mctx, size_t index);
BACKENDS_API void (*RedisAI_ModelRunCtxFree)(RAI_ModelRunCtx *mctx);
BACKENDS_API int (*RedisAI_ModelRun)(RAI_ModelRunCtx **mctx, long long n, RAI_Error *err);
