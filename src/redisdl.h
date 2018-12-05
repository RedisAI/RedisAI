/*
 * redisdl.h
 *
 *  Created on: 2 Dec 2018
 *      Author: root
 */

#ifndef SRC_REDISDL_H_
#define SRC_REDISDL_H_

#include <stdbool.h>
#include "redismodule.h"

#define MODULE_API_FUNC(x) (*x)

typedef struct RDL_Tensor RDL_Tensor;

typedef struct RDL_Graph RDL_Graph;

typedef struct RDL_GraphRunCtx RDL_GraphRunCtx;

RDL_Tensor* MODULE_API_FUNC(RedisDL_TensorCreate)(const char* dataTypeStr, long long* dims,int ndims);
RDL_Tensor* MODULE_API_FUNC(RedisDL_TensorCreateFromTensor)(TF_Tensor *tensor);
size_t MODULE_API_FUNC(RedisDL_TensorGetDataSize)(const char* dataTypeStr);
TF_DataType MODULE_API_FUNC(RedisDL_TensorDataType)(RDL_Tensor* t);
void MODULE_API_FUNC(RedisDL_TensorFree)(RDL_Tensor* t);
int MODULE_API_FUNC(RedisDL_TensorSetData)(RDL_Tensor* tensor, const char* data, size_t len);
int MODULE_API_FUNC(RedisDL_TensorSetValueFromLongLong)(RDL_Tensor* tensor, long long i, long long val);
int MODULE_API_FUNC(RedisDL_TensorSetValueFromDouble)(RDL_Tensor* tensor, long long i, double val);
int MODULE_API_FUNC(RedisDL_TensorGetValueAsDouble)(RDL_Tensor* t, long long i, double* val);
int MODULE_API_FUNC(RedisDL_TensorGetValueAsLongLong)(RDL_Tensor* t, long long i, long long* val);
RDL_Tensor* MODULE_API_FUNC(RedisDL_TensorGetShallowCopy)(RDL_Tensor* t);
int MODULE_API_FUNC(RedisDL_TensorNumDims)(RDL_Tensor* t);
long long MODULE_API_FUNC(RedisDL_TensorDim)(RDL_Tensor* t, int dim);
size_t MODULE_API_FUNC(RedisDL_TensorByteSize)(RDL_Tensor* t);
char* MODULE_API_FUNC(RedisDL_TensorData)(RDL_Tensor* t);

RDL_Graph* MODULE_API_FUNC(RedisDL_GraphCreate)(const char* prefix, const char* graphdef, size_t graphlen);
void MODULE_API_FUNC(RedisDL_GraphFree)(RDL_Graph* graph);
RDL_GraphRunCtx* MODULE_API_FUNC(RedisDL_RunCtxCreate)(RDL_Graph* graph);
int MODULE_API_FUNC(RedisDL_RunCtxAddInput)(RDL_GraphRunCtx* gctx, const char* inputName, RDL_Tensor* inputTensor);
int MODULE_API_FUNC(RedisDL_RunCtxAddOutput)(RDL_GraphRunCtx* gctx, const char* outputName);
size_t MODULE_API_FUNC(RedisDL_RunCtxNumOutputs)(RDL_GraphRunCtx* gctx);
RDL_Tensor* MODULE_API_FUNC(RedisDL_RunCtxOutputTensor)(RDL_GraphRunCtx* gctx, size_t index);
void MODULE_API_FUNC(RedisDL_RunCtxFree)(RDL_GraphRunCtx* gctx);
int MODULE_API_FUNC(RedisDL_GraphRun)(RDL_GraphRunCtx* gctx);
RDL_Graph* MODULE_API_FUNC(RedisDL_GraphGetShallowCopy)(RDL_Graph* graph);

#define REDIDL_MODULE_INIT_FUNCTION(name) \
  if (RedisModule_GetApi("RedisDL_" #name, ((void **)&RedisDL_ ## name))) { \
    printf("could not initialize RedisDL_" #name "\r\n");\
    return false; \
  }

static bool RediDL_Initialize(){
  REDIDL_MODULE_INIT_FUNCTION(TensorCreate);
  REDIDL_MODULE_INIT_FUNCTION(TensorCreateFromTensor);
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


#endif /* SRC_REDISDL_H_ */
