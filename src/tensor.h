#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include "config.h"
#include "tensor_struct.h"
#include "dlpack/dlpack.h"
#include "redismodule.h"

extern RedisModuleType *RedisAI_TensorType;

int RAI_TensorInit(RedisModuleCtx* ctx);
RAI_Tensor* RAI_TensorCreate(const char* dataType, long long* dims, int ndims, int hasdata);
RAI_Tensor* RAI_TensorCreateWithDLDataType(DLDataType dtype, long long* dims, int ndims, int hasdata);
RAI_Tensor* RAI_TensorCreateFromDLTensor(DLManagedTensor* dl_tensor);
RAI_Tensor* RAI_TensorCreateByConcatenatingTensors(RAI_Tensor** ts, long long n);
RAI_Tensor* RAI_TensorCreateBySlicingTensor(RAI_Tensor* t, long long offset, long long len);
size_t RAI_TensorLength(RAI_Tensor* t);
size_t RAI_TensorDataSize(RAI_Tensor* t);
size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype);
size_t RAI_TensorDataSizeFromString(const char* dataType);
DLDataType RAI_TensorDataType(RAI_Tensor* t);
DLDataType RAI_TensorDataTypeFromString(const char* dataType);
void Tensor_DataTypeStr(DLDataType dtype, char **dtypestr);
void RAI_TensorFree(RAI_Tensor* t);
int RAI_TensorSetData(RAI_Tensor* t, const char* data, size_t len);
int RAI_TensorSetValueFromLongLong(RAI_Tensor* t, long long i, long long val);
int RAI_TensorSetValueFromDouble(RAI_Tensor* t, long long i, double val);
int RAI_TensorGetValueAsDouble(RAI_Tensor* t, long long i, double* val);
int RAI_TensorGetValueAsLongLong(RAI_Tensor* t, long long i, long long* val);
RAI_Tensor* RAI_TensorGetShallowCopy(RAI_Tensor* t);
int RAI_TensorNumDims(RAI_Tensor* t);
long long RAI_TensorDim(RAI_Tensor* t, int dim);
size_t RAI_TensorByteSize(RAI_Tensor* t);
char* RAI_TensorData(RAI_Tensor* t);

#endif /* SRC_TENSOR_H_ */
