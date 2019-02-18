#ifndef SRC_REDISAI_H_
#define SRC_REDISAI_H_

#include <stdbool.h>
#include "redismodule.h"

#define MODULE_API_FUNC(x) (*x)

typedef struct RAI_Tensor RAI_Tensor;
typedef struct RAI_Graph RAI_Graph;
typedef struct RAI_Script RAI_Script;

typedef struct RAI_GraphRunCtx RAI_GraphRunCtx;
typedef struct RAI_ScriptRunCtx RAI_ScriptRunCtx;

typedef enum RAI_Backend RAI_Backend;
typedef enum RAI_Device RAI_Device;

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

RAI_Graph* MODULE_API_FUNC(RedisAI_GraphCreate)(RAI_Backend backend, RAI_Device device, const char* graphdef, size_t graphlen);
void MODULE_API_FUNC(RedisAI_GraphFree)(RAI_Graph* graph);
RAI_GraphRunCtx* MODULE_API_FUNC(RedisAI_GraphRunCtxCreate)(RAI_Graph* graph);
int MODULE_API_FUNC(RedisAI_GraphRunCtxAddInput)(RAI_GraphRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor);
int MODULE_API_FUNC(RedisAI_GraphRunCtxAddOutput)(RAI_GraphRunCtx* gctx, const char* outputName);
size_t MODULE_API_FUNC(RedisAI_GraphRunCtxNumOutputs)(RAI_GraphRunCtx* gctx);
RAI_Tensor* MODULE_API_FUNC(RedisAI_GraphRunCtxOutputTensor)(RAI_GraphRunCtx* gctx, size_t index);
void MODULE_API_FUNC(RedisAI_GraphRunCtxFree)(RAI_GraphRunCtx* gctx);
int MODULE_API_FUNC(RedisAI_GraphRun)(RAI_GraphRunCtx* gctx);
RAI_Graph* MODULE_API_FUNC(RedisAI_GraphGetShallowCopy)(RAI_Graph* graph);

RAI_Script* MODULE_API_FUNC(RedisAI_ScriptCreate)(RAI_Backend backend, RAI_Device device, const char* graphdef, size_t graphlen);
void MODULE_API_FUNC(RedisAI_ScriptFree)(RAI_Script* graph);
RAI_ScriptRunCtx* MODULE_API_FUNC(RedisAI_ScriptRunCtxCreate)(RAI_Script* graph);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddInput)(RAI_ScriptRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor);
int MODULE_API_FUNC(RedisAI_ScriptRunCtxAddOutput)(RAI_ScriptRunCtx* gctx, const char* outputName);
size_t MODULE_API_FUNC(RedisAI_ScriptRunCtxNumOutputs)(RAI_ScriptRunCtx* gctx);
RAI_Tensor* MODULE_API_FUNC(RedisAI_ScriptRunCtxOutputTensor)(RAI_ScriptRunCtx* gctx, size_t index);
void MODULE_API_FUNC(RedisAI_ScriptRunCtxFree)(RAI_ScriptRunCtx* gctx);
int MODULE_API_FUNC(RedisAI_ScriptRun)(RAI_ScriptRunCtx* gctx);
RAI_Script* MODULE_API_FUNC(RedisAI_ScriptGetShallowCopy)(RAI_Script* graph);


#define REDISAI_MODULE_INIT_FUNCTION(name) \
  if (RedisModule_GetApi("RedisAI_" #name, ((void **)&RedisAI_ ## name))) { \
    printf("could not initialize RedisAI_" #name "\r\n");\
    return false; \
  }

static bool RediDL_Initialize(){

  RMUTil_InitAlloc();

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

  REDISAI_MODULE_INIT_FUNCTION(GraphCreate);
  REDISAI_MODULE_INIT_FUNCTION(GraphFree);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxCreate);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxAddInput);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxAddOutput);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxNumOutputs);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxOutputTensor);
  REDISAI_MODULE_INIT_FUNCTION(GraphRunCtxFree);
  REDISAI_MODULE_INIT_FUNCTION(GraphRun);
  REDISAI_MODULE_INIT_FUNCTION(GraphGetShallowCopy);

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

  return true;
}

#endif /* SRC_REDISAI_H_ */
