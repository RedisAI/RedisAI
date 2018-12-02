/*
 * tensor.h
 *
 *  Created on: 28 Nov 2018
 *      Author: root
 */

#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include "tensorflow/c/c_api.h"
#include "redismodule.h"

typedef struct RDL_Tensor {
  TF_Tensor* tensor;
  size_t refCount;
}RDL_Tensor;

extern RedisModuleType *RedisDL_TensorType;

int Tensor_Init(RedisModuleCtx* ctx);
RDL_Tensor* Tensor_Create(const char* dataTypeStr, long long* dims,int ndims);
RDL_Tensor* Tensor_CreateFromTensor(TF_Tensor *tensor);
size_t Tensor_GetDataSize(const char* dataTypeStr);
TF_DataType Tensor_DataType(RDL_Tensor* t);
void Tensor_Free(RDL_Tensor* t);
int Tensor_SetData(RDL_Tensor* tensor, const char* data, size_t len);
int Tensor_SetValueFromLongLong(RDL_Tensor* tensor, long long i, long long val);
int Tensor_SetValueFromDouble(RDL_Tensor* tensor, long long i, double val);
int Tensor_GetValueAsDouble(RDL_Tensor* t, long long i, double* val);
int Tensor_GetValueAsLongLong(RDL_Tensor* t, long long i, long long* val);
RDL_Tensor* Tensor_GetShallowCopy(RDL_Tensor* t);



#endif /* SRC_TENSOR_H_ */
