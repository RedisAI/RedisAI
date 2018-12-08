/*
 * tensor.h
 *
 *  Created on: 28 Nov 2018
 *      Author: root
 */

#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include "dlpack/dlpack.h"
#include "redisdl.h"

extern RedisModuleType *RedisDL_TensorType;

int RDL_TensorInit(RedisModuleCtx* ctx);
RDL_Tensor* RDL_TensorCreate(const char* dataTypeStr, long long* dims, int ndims);
size_t RDL_TensorLength(RDL_Tensor* t);
size_t RDL_TensorGetDataSize(const char* dataTypeStr);
DLDataType RDL_TensorDataType(RDL_Tensor* t);
void RDL_TensorFree(RDL_Tensor* t);
int RDL_TensorSetData(RDL_Tensor* t, const char* data, size_t len);
int RDL_TensorSetValueFromLongLong(RDL_Tensor* t, long long i, long long val);
int RDL_TensorSetValueFromDouble(RDL_Tensor* t, long long i, double val);
int RDL_TensorGetValueAsDouble(RDL_Tensor* t, long long i, double* val);
int RDL_TensorGetValueAsLongLong(RDL_Tensor* t, long long i, long long* val);
RDL_Tensor* RDL_TensorGetShallowCopy(RDL_Tensor* t);
int RDL_TensorNumDims(RDL_Tensor* t);
long long RDL_TensorDim(RDL_Tensor* t, int dim);
size_t RDL_TensorByteSize(RDL_Tensor* t);
char* RDL_TensorData(RDL_Tensor* t);

#define RDL_TF_BACKEND
#ifdef RDL_TF_BACKEND
#include "tensorflow/c/c_api.h"

RDL_Tensor* RDL_TensorCreateFromTFTensor(TF_Tensor *tensor);
TF_Tensor* RDL_TFTensorFromTensor(RDL_Tensor* t);
#endif


#endif /* SRC_TENSOR_H_ */
