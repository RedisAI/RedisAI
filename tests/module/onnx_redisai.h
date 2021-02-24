#pragma once

#include <stdbool.h>
#include "redismodule.h"
#include "model_struct.h"

#define REDISAI_LLAPI_VERSION 1
#define MODULE_API_FUNC(x)    (*x)

#ifdef REDISAI_EXTERN
#define REDISAI_API extern
#endif

#ifndef REDISAI_API
#define REDISAI_API
#endif

#ifndef REDISAI_H_INCLUDE
typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Model RAI_Model;
typedef struct RAI_Script RAI_Script;

typedef struct RAI_Error RAI_Error;

#endif

#define REDISAI_BACKEND_TENSORFLOW  0
#define REDISAI_BACKEND_TFLITE      1
#define REDISAI_BACKEND_TORCH       2
#define REDISAI_BACKEND_ONNXRUNTIME 3

#define REDISAI_DEVICE_CPU 0
#define REDISAI_DEVICE_GPU 1

typedef enum RedisAI_ErrorCode {
    RedisAI_ErrorCode_OK = 0,
    RedisAI_ErrorCode_EMODELIMPORT,
    RedisAI_ErrorCode_EMODELCONFIGURE,
    RedisAI_ErrorCode_EMODELCREATE,
    RedisAI_ErrorCode_EMODELRUN,
    RedisAI_ErrorCode_EMODELSERIALIZE,
    RedisAI_ErrorCode_EMODELFREE,
    RedisAI_ErrorCode_ESCRIPTIMPORT,
    RedisAI_ErrorCode_ESCRIPTCONFIGURE,
    RedisAI_ErrorCode_ESCRIPTCREATE,
    RedisAI_ErrorCode_ESCRIPTRUN,
    RedisAI_ErrorCode_EUNSUPPORTEDBACKEND,
    RedisAI_ErrorCode_EBACKENDNOTLOADED,
    RedisAI_ErrorCode_ESCRIPTFREE,
    RedisAI_ErrorCode_ETENSORSET,
    RedisAI_ErrorCode_ETENSORGET,
    RedisAI_ErrorCode_EDAGBUILDER,
    RedisAI_ErrorCode_EDAGRUN,
    RedisAI_ErrorCode_EFINISHCTX
} RedisAI_ErrorCode;

REDISAI_API int MODULE_API_FUNC(RedisAI_LoadDefaultBackend)(RedisModuleCtx *ctx, int backend);
REDISAI_API RAI_Model *MODULE_API_FUNC(RedisAI_ModelCreate)(int backend, char *devicestr, char *tag,
  RAI_ModelOpts opts, size_t ninputs,
  const char **inputs, size_t noutputs,
  const char **outputs,
  const char *modeldef, size_t modellen,
  RAI_Error *err);
REDISAI_API void MODULE_API_FUNC(RedisAI_ModelFree)(RAI_Model *model, RAI_Error *err);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelRun)(RAI_ModelRunCtx **mctx, long long n,
  RAI_Error *err);

REDISAI_API int MODULE_API_FUNC(RedisAI_GetLLAPIVersion)();

#ifndef __cplusplus
#define REDISAI_MODULE_INIT_FUNCTION(ctx, name)                                                    \
    RedisAI_##name = RedisModule_GetSharedAPI(ctx, "RedisAI_" #name);                              \
    if (!RedisAI_##name) {                                                                         \
        RedisModule_Log(ctx, "warning", "could not initialize RedisAI_" #name "\r\n");             \
        return REDISMODULE_ERR;                                                                    \
    }
#else
#define REDISAI_MODULE_INIT_FUNCTION(ctx, name)                                                    \
    RedisAI_##name = reinterpret_cast<decltype(RedisAI_##name)>(                                   \
        RedisModule_GetSharedAPI((RedisModuleCtx *)(ctx), "RedisAI_" #name));                      \
    if (!RedisAI_##name) {                                                                         \
        RedisModule_Log(ctx, "warning", "could not initialize RedisAI_" #name "\r\n");             \
        return REDISMODULE_ERR;                                                                    \
    }
#endif

static int RedisAI_OnnxInitialize(RedisModuleCtx *ctx) {

    if (!RedisModule_GetSharedAPI) {
        RedisModule_Log(ctx, "warning",
                        "redis version is not compatible with module shared api, "
                        "use redis 5.0.4 or above.");
        return REDISMODULE_ERR;
    }

    REDISAI_MODULE_INIT_FUNCTION(ctx, LoadDefaultBackend);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRun);

    REDISAI_MODULE_INIT_FUNCTION(ctx, GetLLAPIVersion);

    if (RedisAI_GetLLAPIVersion() < REDISAI_LLAPI_VERSION) {
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
