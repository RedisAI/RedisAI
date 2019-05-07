#ifndef SRC_REDISAI_H_
#define SRC_REDISAI_H_

#include <stdbool.h>
#include "redismodule.h"

#define REDISAI_LLAPI_VERSION 1

#define MODULE_API_FUNC(x) (*x)

#ifndef REDISAI_H_INCLUDE
typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Model RAI_Model;
typedef struct RAI_Script RAI_Script;

typedef struct RAI_ModelRunCtx RAI_ModelRunCtx;
typedef struct RAI_ScriptRunCtx RAI_ScriptRunCtx;
typedef struct RAI_Error RAI_Error;
#endif

#define REDISAI_BACKEND_TENSORFLOW 0
#define REDISAI_BACKEND_TORCH 1
#define REDISAI_BACKEND_ONNXRUNTIME 2

#define REDISAI_DEVICE_CPU 0
#define REDISAI_DEVICE_GPU 1

RAI_Tensor* MODULE_API_FUNC(RedisAI_TensorCreate)(const char* dataTypeStr, long long* dims, int ndims);
size_t MODULE_API_FUNC(RedisAI_TensorLength)(RAI_Tensor* t);
size_t MODULE_API_FUNC(RedisAI_TensorGetDataSize)(const char* dataTypeStr);
size_t MODULE_API_FUNC(RedisAI_TensorDataType)(RAI_Tensor* t);
void MODULE_API_FUNC(RedisAI_TensorFree)(RAI_Tensor* t);
int MODULE_API_FUNC(RedisAI_TensorSetData)(RAI_Tensor* tensor, const char* data, size_t len);
int MODULE_API_FUNC(RedisAI_TensorSetValueFromLongLong)(RAI_Tensor* tensor, long long i, long long val);
int MODULE_API_FUNC(RedisAI_TensorSetValueFromDouble)(RAI_Tensor* tensor, long long i, double val);
int MODULE_API_FUNC(RedisAI_TensorGetValueAsDouble)(RAI_Tensor* t, long long i, double* val);
int MODULE_API_FUNC(RedisAI_TensorGetValueAsLongLong)(RAI_Tensor* t, long long i, long long* val);
RAI_Tensor* MODULE_API_FUNC(RedisAI_TensorGetShallowCopy)(RAI_Tensor* t);
int MODULE_API_FUNC(RedisAI_TensorNumDims)(RAI_Tensor* t);
long long MODULE_API_FUNC(RedisAI_TensorDim)(RAI_Tensor* t, int dim);
size_t MODULE_API_FUNC(RedisAI_TensorByteSize)(RAI_Tensor* t);
char* MODULE_API_FUNC(RedisAI_TensorData)(RAI_Tensor* t);

RAI_Model* MODULE_API_FUNC(RedisAI_ModelCreate)(int backend, int device,
                                                size_t ninputs, const char **inputs,
                                                size_t noutputs, const char **outputs,
                                                const char *modeldef, size_t modellen, RAI_Error* err);
void MODULE_API_FUNC(RedisAI_ModelFree)(RAI_Model* model, RAI_Error* err);
RAI_ModelRunCtx* MODULE_API_FUNC(RedisAI_ModelRunCtxCreate)(RAI_Model* model);
int MODULE_API_FUNC(RedisAI_ModelRunCtxAddInput)(RAI_ModelRunCtx* mctx, const char* inputName, RAI_Tensor* inputTensor);
int MODULE_API_FUNC(RedisAI_ModelRunCtxAddOutput)(RAI_ModelRunCtx* mctx, const char* outputName);
size_t MODULE_API_FUNC(RedisAI_ModelRunCtxNumOutputs)(RAI_ModelRunCtx* mctx);
RAI_Tensor* MODULE_API_FUNC(RedisAI_ModelRunCtxOutputTensor)(RAI_ModelRunCtx* mctx, size_t index);
void MODULE_API_FUNC(RedisAI_ModelRunCtxFree)(RAI_ModelRunCtx* mctx);
int MODULE_API_FUNC(RedisAI_ModelRun)(RAI_ModelRunCtx* mctx, RAI_Error* err);
RAI_Model* MODULE_API_FUNC(RedisAI_ModelGetShallowCopy)(RAI_Model* model);
int MODULE_API_FUNC(RedisAI_ModelSerialize)(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err);

RAI_Script* MODULE_API_FUNC(RedisAI_ScriptCreate)(int device, const char* scriptdef, RAI_Error* err);
void MODULE_API_FUNC(RedisAI_ScriptFree)(RAI_Script* script, RAI_Error* err);
RAI_ScriptRunCtx* MODULE_API_FUNC(RedisAI_ScriptRunCtxCreate)(RAI_Script* script, const char *fnname);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInput)(RAI_ScriptRunCtx* sctx, RAI_Tensor* inputTensor);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddOutput)(RAI_ScriptRunCtx* sctx);
size_t MODULE_API_FUNC(RedisAI_ScriptRunCtxNumOutputs)(RAI_ScriptRunCtx* sctx);
RAI_Tensor* MODULE_API_FUNC(RedisAI_ScriptRunCtxOutputTensor)(RAI_ScriptRunCtx* sctx, size_t index);
void MODULE_API_FUNC(RedisAI_ScriptRunCtxFree)(RAI_ScriptRunCtx* sctx);
int MODULE_API_FUNC(RedisAI_ScriptRun)(RAI_ScriptRunCtx* sctx, RAI_Error* err);
RAI_Script* MODULE_API_FUNC(RedisAI_ScriptGetShallowCopy)(RAI_Script* script);

int MODULE_API_FUNC(RedisAI_GetLLAPIVersion)();


#define REDISAI_MODULE_INIT_FUNCTION(name) \
  if (RedisModule_GetApi("RedisAI_" #name, ((void **)&RedisAI_ ## name))) { \
    printf("could not initialize RedisAI_" #name "\r\n");\
    return REDISMODULE_ERR; \
  }

static int RedisAI_Initialize(){

  REDISAI_MODULE_INIT_FUNCTION(GetLLAPIVersion);

  REDISAI_MODULE_INIT_FUNCTION(TensorCreate);
  REDISAI_MODULE_INIT_FUNCTION(TensorGetDataSize);
  REDISAI_MODULE_INIT_FUNCTION(TensorFree);
  REDISAI_MODULE_INIT_FUNCTION(TensorSetData);
  REDISAI_MODULE_INIT_FUNCTION(TensorSetValueFromLongLong);
  REDISAI_MODULE_INIT_FUNCTION(TensorSetValueFromDouble);
  REDISAI_MODULE_INIT_FUNCTION(TensorGetValueAsDouble);
  REDISAI_MODULE_INIT_FUNCTION(TensorGetValueAsLongLong);
  REDISAI_MODULE_INIT_FUNCTION(TensorGetShallowCopy);
  REDISAI_MODULE_INIT_FUNCTION(TensorNumDims);
  REDISAI_MODULE_INIT_FUNCTION(TensorDim);
  REDISAI_MODULE_INIT_FUNCTION(TensorByteSize);
  REDISAI_MODULE_INIT_FUNCTION(TensorData);

  REDISAI_MODULE_INIT_FUNCTION(ModelCreate);
  REDISAI_MODULE_INIT_FUNCTION(ModelFree);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxCreate);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxAddInput);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxAddOutput);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxNumOutputs);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxOutputTensor);
  REDISAI_MODULE_INIT_FUNCTION(ModelRunCtxFree);
  REDISAI_MODULE_INIT_FUNCTION(ModelRun);
  REDISAI_MODULE_INIT_FUNCTION(ModelGetShallowCopy);
  REDISAI_MODULE_INIT_FUNCTION(ModelSerialize);

  REDISAI_MODULE_INIT_FUNCTION(ScriptCreate);
  REDISAI_MODULE_INIT_FUNCTION(ScriptFree);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxCreate);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxAddInput);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxAddOutput);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxNumOutputs);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxOutputTensor);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRunCtxFree);
  REDISAI_MODULE_INIT_FUNCTION(ScriptRun);
  REDISAI_MODULE_INIT_FUNCTION(ScriptGetShallowCopy);

  if(RedisAI_GetLLAPIVersion() < REDISAI_LLAPI_VERSION){
    return REDISMODULE_ERR;
  }

  return REDISMODULE_OK;
}

#endif /* SRC_REDISAI_H_ */
