#ifndef SRC_REDISAI_H_
#define SRC_REDISAI_H_

#include <stdbool.h>
#include "redismodule.h"

#define MODULE_API_FUNC(x) (*x)

typedef struct RAI_Tensor RAI_Tensor;

typedef struct RAI_Graph RAI_Graph;

typedef struct RAI_GraphRunCtx RAI_GraphRunCtx;

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

RAI_Graph* MODULE_API_FUNC(RedisAI_GraphCreate)(const char* prefix, RAI_Backend backend, RAI_Device device, const char* graphdef, size_t graphlen);
void MODULE_API_FUNC(RedisAI_GraphFree)(RAI_Graph* graph);
RAI_GraphRunCtx* MODULE_API_FUNC(RedisAI_RunCtxCreate)(RAI_Graph* graph);
int MODULE_API_FUNC(RedisAI_RunCtxAddInput)(RAI_GraphRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor);
int MODULE_API_FUNC(RedisAI_RunCtxAddOutput)(RAI_GraphRunCtx* gctx, const char* outputName);
size_t MODULE_API_FUNC(RedisAI_RunCtxNumOutputs)(RAI_GraphRunCtx* gctx);
RAI_Tensor* MODULE_API_FUNC(RedisAI_RunCtxOutputTensor)(RAI_GraphRunCtx* gctx, size_t index);
void MODULE_API_FUNC(RedisAI_RunCtxFree)(RAI_GraphRunCtx* gctx);
int MODULE_API_FUNC(RedisAI_GraphRun)(RAI_GraphRunCtx* gctx);
RAI_Graph* MODULE_API_FUNC(RedisAI_GraphGetShallowCopy)(RAI_Graph* graph);

#define REDIDL_MODULE_INIT_FUNCTION(name) \
  if (RedisModule_GetApi("RedisAI_" #name, ((void **)&RedisAI_ ## name))) { \
    printf("could not initialize RedisAI_" #name "\r\n");\
    return false; \
  }

static bool RediDL_Initialize(){
  REDIDL_MODULE_INIT_FUNCTION(TensorCreate);
  REDIDL_MODULE_INIT_FUNCTION(TensorLength);
  REDIDL_MODULE_INIT_FUNCTION(TensorGetDataSize);
  REDIDL_MODULE_INIT_FUNCTION(TensorDataType);
  REDIDL_MODULE_INIT_FUNCTION(TensorFree);
  REDIDL_MODULE_INIT_FUNCTION(TensorSetData);
  REDIDL_MODULE_INIT_FUNCTION(TensorSetValueFromLongLong);
  REDIDL_MODULE_INIT_FUNCTION(TensorSetValueFromDouble);
  REDIDL_MODULE_INIT_FUNCTION(TensorGetValueAsDouble);
  REDIDL_MODULE_INIT_FUNCTION(TensorGetValueAsLongLong);
  REDIDL_MODULE_INIT_FUNCTION(TensorGetShallowCopy);
  REDIDL_MODULE_INIT_FUNCTION(TensorNumDims);
  REDIDL_MODULE_INIT_FUNCTION(TensorDim);
  REDIDL_MODULE_INIT_FUNCTION(TensorByteSize);
  REDIDL_MODULE_INIT_FUNCTION(TensorData);

  REDIDL_MODULE_INIT_FUNCTION(GraphCreate);
  REDIDL_MODULE_INIT_FUNCTION(GraphFree);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxCreate);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxAddInput);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxAddOutput);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxNumOutputs);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxOutputTensor);
  REDIDL_MODULE_INIT_FUNCTION(RunCtxFree);
  REDIDL_MODULE_INIT_FUNCTION(GraphRun);
  REDIDL_MODULE_INIT_FUNCTION(GraphGetShallowCopy);
  return true;
}

#endif /* SRC_REDISAI_H_ */
