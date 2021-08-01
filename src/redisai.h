#pragma once

#include <stdbool.h>
#include "redismodule.h"

#define REDISAI_LLAPI_VERSION 1
#define MODULE_API_FUNC(x)    (*x)

#ifdef REDISAI_MAIN
#define REDISAI_API 
#endif

#ifndef REDISAI_API
#define REDISAI_API extern
#endif

#ifndef REDISAI_H_INCLUDE
typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Model RAI_Model;
typedef struct RAI_Script RAI_Script;

typedef struct RAI_ModelRunCtx RAI_ModelRunCtx;
typedef struct RAI_ScriptRunCtx RAI_ScriptRunCtx;
typedef struct RAI_DAGRunCtx RAI_DAGRunCtx;
typedef struct RAI_DAGRunOp RAI_DAGRunOp;
typedef struct RAI_Error RAI_Error;
typedef struct RAI_ModelOpts RAI_ModelOpts;
typedef struct RAI_OnFinishCtx RAI_OnFinishCtx;

typedef void (*RAI_OnFinishCB)(RAI_OnFinishCtx *ctx, void *private_data);
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

REDISAI_API int MODULE_API_FUNC(RedisAI_InitError)(RAI_Error **err);
REDISAI_API void MODULE_API_FUNC(RedisAI_ClearError)(RAI_Error *err);
REDISAI_API void MODULE_API_FUNC(RedisAI_FreeError)(RAI_Error *err);
REDISAI_API const char *MODULE_API_FUNC(RedisAI_GetError)(RAI_Error *err);
REDISAI_API const char *MODULE_API_FUNC(RedisAI_GetErrorOneLine)(RAI_Error *err);
REDISAI_API RedisAI_ErrorCode MODULE_API_FUNC(RedisAI_GetErrorCode)(RAI_Error *err);
REDISAI_API void MODULE_API_FUNC(RedisAI_CloneError)(RAI_Error *dest, const RAI_Error *src);

REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreate)(const char *dataTypeStr,
                                                              long long *dims, int ndims);
REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreateByConcatenatingTensors)(RAI_Tensor **ts,
                                                                                    long long n);
REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreateBySlicingTensor)(RAI_Tensor *t,
                                                                             long long offset,
                                                                             long long len);

REDISAI_API size_t MODULE_API_FUNC(RedisAI_TensorLength)(RAI_Tensor *t);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_TensorDataSize)(RAI_Tensor *t);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_TensorDataType)(RAI_Tensor *t);
REDISAI_API void MODULE_API_FUNC(RedisAI_TensorFree)(RAI_Tensor *t);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorSetData)(RAI_Tensor *tensor, const char *data,
                                                       size_t len);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorSetValueFromLongLong)(RAI_Tensor *tensor, long long i,
                                                                    long long val);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorSetValueFromDouble)(RAI_Tensor *tensor, long long i,
                                                                  double val);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorGetValueAsDouble)(RAI_Tensor *t, long long i,
                                                                double *val);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorGetValueAsLongLong)(RAI_Tensor *t, long long i,
                                                                  long long *val);
REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorGetShallowCopy)(RAI_Tensor *t);
REDISAI_API int MODULE_API_FUNC(RedisAI_TensorNumDims)(RAI_Tensor *t);
REDISAI_API long long MODULE_API_FUNC(RedisAI_TensorDim)(RAI_Tensor *t, int dim);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_TensorByteSize)(RAI_Tensor *t);
REDISAI_API char *MODULE_API_FUNC(RedisAI_TensorData)(RAI_Tensor *t);
REDISAI_API RedisModuleType *MODULE_API_FUNC(RedisAI_TensorRedisType)(void);

REDISAI_API RAI_Model *MODULE_API_FUNC(RedisAI_ModelCreate)(int backend, char *devicestr, char *tag,
                                                            RAI_ModelOpts opts, size_t ninputs,
                                                            const char **inputs, size_t noutputs,
                                                            const char **outputs,
                                                            const char *modeldef, size_t modellen,
                                                            RAI_Error *err);
REDISAI_API void MODULE_API_FUNC(RedisAI_ModelFree)(RAI_Model *model, RAI_Error *err);
REDISAI_API RAI_ModelRunCtx *MODULE_API_FUNC(RedisAI_ModelRunCtxCreate)(RAI_Model *model);
REDISAI_API int MODULE_API_FUNC(RedisAI_GetModelFromKeyspace)(RedisModuleCtx *ctx,
                                                              RedisModuleString *keyName,
                                                              RAI_Model **model, int mode,
                                                              RAI_Error *err);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelRunCtxAddInput)(RAI_ModelRunCtx *mctx,
                                                             const char *inputName,
                                                             RAI_Tensor *inputTensor);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelRunCtxAddOutput)(RAI_ModelRunCtx *mctx,
                                                              const char *outputName);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_ModelRunCtxNumOutputs)(RAI_ModelRunCtx *mctx);
REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_ModelRunCtxOutputTensor)(RAI_ModelRunCtx *mctx,
                                                                         size_t index);
REDISAI_API void MODULE_API_FUNC(RedisAI_ModelRunCtxFree)(RAI_ModelRunCtx *mctx);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelRun)(RAI_ModelRunCtx **mctx, long long n,
                                                  RAI_Error *err);
REDISAI_API RAI_Model *MODULE_API_FUNC(RedisAI_ModelGetShallowCopy)(RAI_Model *model);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelSerialize)(RAI_Model *model, char **buffer,
                                                        size_t *len, RAI_Error *err);
REDISAI_API RedisModuleType *MODULE_API_FUNC(RedisAI_ModelRedisType)(void);
REDISAI_API int MODULE_API_FUNC(RedisAI_ModelRunAsync)(RAI_ModelRunCtx *mctxs,
                                                       RAI_OnFinishCB DAGAsyncFinish,
                                                       void *private_data);
REDISAI_API RAI_ModelRunCtx *MODULE_API_FUNC(RedisAI_GetAsModelRunCtx)(RAI_OnFinishCtx *ctx,
                                                                       RAI_Error *err);

REDISAI_API RAI_Script *MODULE_API_FUNC(RedisAI_ScriptCreate)(char *devicestr, char *tag,
                                                              const char *scriptdef,
                                                              RAI_Error *err);
REDISAI_API int MODULE_API_FUNC(RedisAI_GetScriptFromKeyspace)(RedisModuleCtx *ctx,
                                                               RedisModuleString *keyName,
                                                               RAI_Script **script, int mode,
                                                               RAI_Error *err);
REDISAI_API void MODULE_API_FUNC(RedisAI_ScriptFree)(RAI_Script *script, RAI_Error *err);
REDISAI_API RAI_ScriptRunCtx *MODULE_API_FUNC(RedisAI_ScriptRunCtxCreate)(RAI_Script *script,
                                                                          const char *fnname);
// Deprecated, use RedisAI_ScriptRunCtxAddInputTensor instead.
REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInput)(RAI_ScriptRunCtx *sctx,
                                                              RAI_Tensor *inputTensor,
                                                              RAI_Error *err);

// Deprecated, use RedisAI_ScriptRunCtxAddTensorInputList instead.
REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInputList)(RAI_ScriptRunCtx *sctx,
                                                                  RAI_Tensor **inputTensors,
                                                                  size_t len, RAI_Error *err);

REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddTensorInput)(RAI_ScriptRunCtx *sctx,
                                                                    RAI_Tensor *inputTensor);

REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddTensorInputList)(RAI_ScriptRunCtx *sctx,
                                                                        RAI_Tensor **inputTensors,
                                                                        size_t count);

REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddOutput)(RAI_ScriptRunCtx *sctx);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_ScriptRunCtxNumOutputs)(RAI_ScriptRunCtx *sctx);
REDISAI_API RAI_Tensor *MODULE_API_FUNC(RedisAI_ScriptRunCtxOutputTensor)(RAI_ScriptRunCtx *sctx,
                                                                          size_t index);
REDISAI_API void MODULE_API_FUNC(RedisAI_ScriptRunCtxFree)(RAI_ScriptRunCtx *sctx);
REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRun)(RAI_ScriptRunCtx *sctx, RAI_Error *err);
REDISAI_API RAI_Script *MODULE_API_FUNC(RedisAI_ScriptGetShallowCopy)(RAI_Script *script);
REDISAI_API RedisModuleType *MODULE_API_FUNC(RedisAI_ScriptRedisType)(void);
REDISAI_API int MODULE_API_FUNC(RedisAI_ScriptRunAsync)(RAI_ScriptRunCtx *sctx,
                                                        RAI_OnFinishCB DAGAsyncFinish,
                                                        void *private_data);
REDISAI_API RAI_ScriptRunCtx *MODULE_API_FUNC(RedisAI_GetAsScriptRunCtx)(RAI_OnFinishCtx *ctx,
                                                                         RAI_Error *err);

REDISAI_API RAI_DAGRunCtx *MODULE_API_FUNC(RedisAI_DAGRunCtxCreate)(void);
REDISAI_API RAI_DAGRunOp *MODULE_API_FUNC(RedisAI_DAGCreateModelRunOp)(RAI_Model *model);
REDISAI_API RAI_DAGRunOp *MODULE_API_FUNC(RedisAI_DAGCreateScriptRunOp)(RAI_Script *script,
                                                                        const char *func_name);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGRunOpAddInput)(RAI_DAGRunOp *DAGOp, const char *input);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGRunOpAddOutput)(RAI_DAGRunOp *DAGOp, const char *output);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGAddRunOp)(RAI_DAGRunCtx *run_info, RAI_DAGRunOp *DAGop,
                                                     RAI_Error *err);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGLoadTensor)(RAI_DAGRunCtx *run_info, const char *t_name,
                                                       RAI_Tensor *tensor);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGAddTensorSet)(RAI_DAGRunCtx *run_info,
                                                         const char *t_name, RAI_Tensor *tensor);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGAddTensorGet)(RAI_DAGRunCtx *run_info,
                                                         const char *t_name);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGAddOpsFromString)(RAI_DAGRunCtx *run_info,
                                                             const char *dag, RAI_Error *err);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_DAGNumOps)(RAI_DAGRunCtx *run_info);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGRun)(RAI_DAGRunCtx *run_info,
                                                RAI_OnFinishCB DAGAsyncFinish, void *private_data,
                                                RAI_Error *err);
REDISAI_API size_t MODULE_API_FUNC(RedisAI_DAGNumOutputs)(RAI_OnFinishCtx *finish_ctx);
REDISAI_API const RAI_Tensor *MODULE_API_FUNC(RedisAI_DAGOutputTensor)(RAI_OnFinishCtx *finish_ctx,
                                                                       size_t index);
REDISAI_API int MODULE_API_FUNC(RedisAI_DAGRunError)(RAI_OnFinishCtx *finish_ctx);
REDISAI_API const RAI_Error *MODULE_API_FUNC(RedisAI_DAGGetError)(RAI_OnFinishCtx *finish_ctx);
REDISAI_API void MODULE_API_FUNC(RedisAI_DAGRunOpFree)(RAI_DAGRunOp *dagOp);
REDISAI_API void MODULE_API_FUNC(RedisAI_DAGFree)(RAI_DAGRunCtx *run_info);

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

static int RedisAI_Initialize(RedisModuleCtx *ctx) {

    if (!RedisModule_GetSharedAPI) {
        RedisModule_Log(ctx, "warning",
                        "redis version is not compatible with module shared api, "
                        "use redis 5.0.4 or above.");
        return REDISMODULE_ERR;
    }

    REDISAI_MODULE_INIT_FUNCTION(ctx, InitError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ClearError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, FreeError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetErrorOneLine);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetErrorCode);
    REDISAI_MODULE_INIT_FUNCTION(ctx, CloneError);

    REDISAI_MODULE_INIT_FUNCTION(ctx, GetLLAPIVersion);

    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorCreateByConcatenatingTensors);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorCreateBySlicingTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorLength);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorDataSize);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorSetData);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorSetValueFromLongLong);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorSetValueFromDouble);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorGetValueAsDouble);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorGetValueAsLongLong);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorGetShallowCopy);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorNumDims);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorDim);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorByteSize);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorData);
    REDISAI_MODULE_INIT_FUNCTION(ctx, TensorRedisType);

    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetModelFromKeyspace);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxAddInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxAddOutput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxNumOutputs);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxOutputTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRun);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelGetShallowCopy);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelSerialize);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRedisType);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunAsync);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetAsModelRunCtx);

    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetScriptFromKeyspace);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddTensorInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddInputList);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddTensorInputList);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddOutput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxNumOutputs);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxOutputTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRun);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptGetShallowCopy);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRedisType);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunAsync);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetAsScriptRunCtx);

    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRunCtxCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGCreateModelRunOp);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGCreateScriptRunOp);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRunOpAddInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRunOpAddOutput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGAddRunOp);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGLoadTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGAddTensorSet);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGAddTensorGet);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGAddOpsFromString);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGNumOps);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRun);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGNumOutputs);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGOutputTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRunError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGGetError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGRunOpFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, DAGFree);

    if (RedisAI_GetLLAPIVersion() < REDISAI_LLAPI_VERSION) {
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
