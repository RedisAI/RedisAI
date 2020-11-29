#ifndef SRC_REDISAI_H_
#define SRC_REDISAI_H_

#include <stdbool.h>
#include "redismodule.h"
#include "util/dict.h"
#include "model_struct.h"
#include "background_workers.h"
#include "version.h"

#define MODULE_API_FUNC(x) (*x)

#ifndef REDISAI_H_INCLUDE
typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Model RAI_Model;
typedef struct RAI_Script RAI_Script;

typedef struct RAI_ModelRunCtx RAI_ModelRunCtx;
typedef struct RAI_ScriptRunCtx RAI_ScriptRunCtx;
typedef struct RAI_Error RAI_Error;
#endif

#define REDISAI_BACKEND_TENSORFLOW  0
#define REDISAI_BACKEND_TFLITE      1
#define REDISAI_BACKEND_TORCH       2
#define REDISAI_BACKEND_ONNXRUNTIME 3

#define REDISAI_DEVICE_CPU 0
#define REDISAI_DEVICE_GPU 1

#define RedisAI_ErrorCode                     int
#define RedisAI_ErrorCode_OK                  0
#define RedisAI_ErrorCode_EMODELIMPORT        1
#define RedisAI_ErrorCode_EMODELCONFIGURE     2
#define RedisAI_ErrorCode_EMODELCREATE        3
#define RedisAI_ErrorCode_EMODELRUN           4
#define RedisAI_ErrorCode_EMODELSERIALIZE     5
#define RedisAI_ErrorCode_EMODELFREE          6
#define RedisAI_ErrorCode_ESCRIPTIMPORT       7
#define RedisAI_ErrorCode_ESCRIPTCONFIGURE    8
#define RedisAI_ErrorCode_ESCRIPTCREATE       9
#define RedisAI_ErrorCode_ESCRIPTRUN          10
#define RedisAI_ErrorCode_EUNSUPPORTEDBACKEND 11
#define RedisAI_ErrorCode_EBACKENDNOTLOADED   12
#define RedisAI_ErrorCode_ESCRIPTFREE         13
#define RedisAI_ErrorCode_ETENSORSET          14
#define RedisAI_ErrorCode_ETENSORGET          15
#define RedisAI_ErrorCode_EDAGRUN             17

enum RedisAI_DataFmt { REDISAI_DATA_BLOB = 0, REDISAI_DATA_VALUES, REDISAI_DATA_NONE };

int MODULE_API_FUNC(RedisAI_InitError)(RAI_Error **err);
void MODULE_API_FUNC(RedisAI_ClearError)(RAI_Error *err);
void MODULE_API_FUNC(RedisAI_FreeError)(RAI_Error *err);
const char *MODULE_API_FUNC(RedisAI_GetError)(RAI_Error *err);
const char *MODULE_API_FUNC(RedisAI_GetErrorOneLine)(RAI_Error *err);
RedisAI_ErrorCode MODULE_API_FUNC(RedisAI_GetErrorCode)(RAI_Error *err);

RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreate)(const char *dataTypeStr, long long *dims,
                                                  int ndims);
RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreateByConcatenatingTensors)(RAI_Tensor **ts,
                                                                        long long n);
RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorCreateBySlicingTensor)(RAI_Tensor *t, long long offset,
                                                                 long long len);
size_t MODULE_API_FUNC(RedisAI_TensorLength)(RAI_Tensor *t);
size_t MODULE_API_FUNC(RedisAI_TensorDataSize)(RAI_Tensor *t);
size_t MODULE_API_FUNC(RedisAI_TensorDataType)(RAI_Tensor *t);
void MODULE_API_FUNC(RedisAI_TensorFree)(RAI_Tensor *t);
int MODULE_API_FUNC(RedisAI_TensorSetData)(RAI_Tensor *tensor, const char *data, size_t len);
int MODULE_API_FUNC(RedisAI_TensorSetValueFromLongLong)(RAI_Tensor *tensor, long long i,
                                                        long long val);
int MODULE_API_FUNC(RedisAI_TensorSetValueFromDouble)(RAI_Tensor *tensor, long long i, double val);
int MODULE_API_FUNC(RedisAI_TensorGetValueAsDouble)(RAI_Tensor *t, long long i, double *val);
int MODULE_API_FUNC(RedisAI_TensorGetValueAsLongLong)(RAI_Tensor *t, long long i, long long *val);
RAI_Tensor *MODULE_API_FUNC(RedisAI_TensorGetShallowCopy)(RAI_Tensor *t);
int MODULE_API_FUNC(RedisAI_TensorNumDims)(RAI_Tensor *t);
long long MODULE_API_FUNC(RedisAI_TensorDim)(RAI_Tensor *t, int dim);
size_t MODULE_API_FUNC(RedisAI_TensorByteSize)(RAI_Tensor *t);
char *MODULE_API_FUNC(RedisAI_TensorData)(RAI_Tensor *t);
RedisModuleType *MODULE_API_FUNC(RedisAI_TensorRedisType)(void);

RAI_Model *MODULE_API_FUNC(RedisAI_ModelCreate)(int backend, char *devicestr, char *tag,
                                                RAI_ModelOpts opts, size_t ninputs,
                                                const char **inputs, size_t noutputs,
                                                const char **outputs, const char *modeldef,
                                                size_t modellen, RAI_Error *err);
void MODULE_API_FUNC(RedisAI_ModelFree)(RAI_Model *model, RAI_Error *err);
RAI_ModelRunCtx *MODULE_API_FUNC(RedisAI_ModelRunCtxCreate)(RAI_Model *model);
int MODULE_API_FUNC(RedisAI_ModelRunCtxAddInput)(RAI_ModelRunCtx *mctx, const char *inputName,
                                                 RAI_Tensor *inputTensor);
int MODULE_API_FUNC(RedisAI_ModelRunCtxAddOutput)(RAI_ModelRunCtx *mctx, const char *outputName);
size_t MODULE_API_FUNC(RedisAI_ModelRunCtxNumOutputs)(RAI_ModelRunCtx *mctx);
RAI_Tensor *MODULE_API_FUNC(RedisAI_ModelRunCtxOutputTensor)(RAI_ModelRunCtx *mctx, size_t index);
void MODULE_API_FUNC(RedisAI_ModelRunCtxFree)(RAI_ModelRunCtx *mctx);
int MODULE_API_FUNC(RedisAI_ModelRun)(RAI_ModelRunCtx **mctx, long long n, RAI_Error *err);
RAI_Model *MODULE_API_FUNC(RedisAI_ModelGetShallowCopy)(RAI_Model *model);
int MODULE_API_FUNC(RedisAI_ModelSerialize)(RAI_Model *model, char **buffer, size_t *len,
                                            RAI_Error *err);
RedisModuleType *MODULE_API_FUNC(RedisAI_ModelRedisType)(void);

RAI_Script *MODULE_API_FUNC(RedisAI_ScriptCreate)(char *devicestr, char *tag, const char *scriptdef,
                                                  RAI_Error *err);
void MODULE_API_FUNC(RedisAI_ScriptFree)(RAI_Script *script, RAI_Error *err);
RAI_ScriptRunCtx *MODULE_API_FUNC(RedisAI_ScriptRunCtxCreate)(RAI_Script *script,
                                                              const char *fnname);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInput)(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor,
                                                  RAI_Error *err);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInputList)(RAI_ScriptRunCtx *sctx,
                                                      RAI_Tensor **inputTensors, size_t len,
                                                      RAI_Error *err);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddOutput)(RAI_ScriptRunCtx *sctx);
size_t MODULE_API_FUNC(RedisAI_ScriptRunCtxNumOutputs)(RAI_ScriptRunCtx *sctx);
RAI_Tensor *MODULE_API_FUNC(RedisAI_ScriptRunCtxOutputTensor)(RAI_ScriptRunCtx *sctx, size_t index);
void MODULE_API_FUNC(RedisAI_ScriptRunCtxFree)(RAI_ScriptRunCtx *sctx);
int MODULE_API_FUNC(RedisAI_ScriptRun)(RAI_ScriptRunCtx *sctx, RAI_Error *err);
RAI_Script *MODULE_API_FUNC(RedisAI_ScriptGetShallowCopy)(RAI_Script *script);
RedisModuleType *MODULE_API_FUNC(RedisAI_ScriptRedisType)(void);

int MODULE_API_FUNC(RedisAI_GetLLAPIVersion)();

#define REDISAI_MODULE_INIT_FUNCTION(ctx, name)                                                    \
    RedisAI_##name = RedisModule_GetSharedAPI(ctx, "RedisAI_" #name);                              \
    if (!RedisAI_##name) {                                                                         \
        RedisModule_Log(ctx, "warning", "could not initialize RedisAI_" #name "\r\n");             \
        return REDISMODULE_ERR;                                                                    \
    }

static int RedisAI_Initialize(RedisModuleCtx *ctx) {

    if (!RedisModule_GetSharedAPI) {
        RedisModule_Log(
            ctx, "warning",
            "redis version is not compatible with module shared api, use redis 5.0.4 or above.");
        return REDISMODULE_ERR;
    }

    REDISAI_MODULE_INIT_FUNCTION(ctx, InitError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ClearError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, FreeError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetError);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetErrorOneLine);
    REDISAI_MODULE_INIT_FUNCTION(ctx, GetErrorCode);

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
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxAddInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxAddOutput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxNumOutputs);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxOutputTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRunCtxFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRun);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelGetShallowCopy);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelSerialize);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ModelRedisType);

    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxCreate);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddInput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddInputList);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxAddOutput);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxNumOutputs);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxOutputTensor);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRunCtxFree);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRun);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptGetShallowCopy);
    REDISAI_MODULE_INIT_FUNCTION(ctx, ScriptRedisType);

    if (RedisAI_GetLLAPIVersion() < REDISAI_LLAPI_VERSION) {
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}

#endif /* SRC_REDISAI_H_ */
