#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include "config.h"
#include "tensor_struct.h"
#include "dlpack/dlpack.h"
#include "redismodule.h"
#include "util/dict.h"

#define TENSORALLOC_NONE 0
#define TENSORALLOC_ALLOC 1
#define TENSORALLOC_CALLOC 2

// Numeric data type of tensor elements, one of FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16
static const char* RAI_DATATYPE_STR_FLOAT = "FLOAT";
static const char* RAI_DATATYPE_STR_DOUBLE = "DOUBLE";
static const char* RAI_DATATYPE_STR_INT8 = "INT8";
static const char* RAI_DATATYPE_STR_INT16 = "INT16";
static const char* RAI_DATATYPE_STR_INT32 = "INT32";
static const char* RAI_DATATYPE_STR_INT64 = "INT64";
static const char* RAI_DATATYPE_STR_UINT8 = "UINT8";
static const char* RAI_DATATYPE_STR_UINT16 = "UINT16";

extern RedisModuleType *RedisAI_TensorType;

int RAI_TensorInit(RedisModuleCtx* ctx);
RAI_Tensor* RAI_TensorCreate(const char* dataType, long long* dims, int ndims, int hasdata);
RAI_Tensor* RAI_TensorCreateWithDLDataType(DLDataType dtype, long long* dims, int ndims, int tensorAllocMode);
RAI_Tensor* RAI_TensorCreateFromDLTensor(DLManagedTensor* dl_tensor);
RAI_Tensor* RAI_TensorCreateByConcatenatingTensors(RAI_Tensor** ts, long long n);
RAI_Tensor* RAI_TensorCreateBySlicingTensor(RAI_Tensor* t, long long offset, long long len);
size_t RAI_TensorLength(RAI_Tensor* t);
size_t RAI_TensorDataSize(RAI_Tensor* t);
size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype);
size_t RAI_TensorDataSizeFromString(const char* dataType);
DLDataType RAI_TensorDataType(RAI_Tensor* t);
DLDataType RAI_TensorDataTypeFromString(const char* dataType);
int Tensor_DataTypeStr(DLDataType dtype, char **dtypestr);
void RAI_TensorFree(RAI_Tensor* t);
int RAI_TensorSetData(RAI_Tensor* t, const char* data, size_t len);
int RAI_TensorSetDataFromRS(RAI_Tensor* t, RedisModuleString* rs);
int RAI_TensorSetValueFromLongLong(RAI_Tensor* t, long long i, long long val);
int RAI_TensorSetValueFromDouble(RAI_Tensor* t, long long i, double val);
int RAI_TensorGetValueAsDouble(RAI_Tensor* t, long long i, double* val);
int RAI_TensorGetValueAsLongLong(RAI_Tensor* t, long long i, long long* val);
RAI_Tensor* RAI_TensorGetShallowCopy(RAI_Tensor* t);
int RAI_TensorNumDims(RAI_Tensor* t);
long long RAI_TensorDim(RAI_Tensor* t, int dim);
size_t RAI_TensorByteSize(RAI_Tensor* t);
char* RAI_TensorData(RAI_Tensor* t);

/* Return REDISMODULE_ERR if is the key not associated with a tensor type.
 * Return REDISMODULE_OK otherwise. */
int RAI_OpenKey_Tensor(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key,
                              int mode);

/* Return REDISMODULE_ERR if there was an error getting the Tensor.
 * Return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable. */
int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                               RedisModuleKey **key, RAI_Tensor **tensor,
                               int mode);

/* Return REDISMODULE_ERR if there was an error getting the Tensor.
 * Return REDISMODULE_OK if the tensor value is present at the localContextDict. */
int RAI_getTensorFromLocalContext(RedisModuleCtx *ctx,
                                  AI_dict *localContextDict,
                                  const char *localContextKey,
                                  RAI_Tensor **tensor);

void RedisAI_ReplicateTensorSet(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t);

int RAI_parseTensorSetArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, RAI_Tensor **t, int enforceArity);

int RAI_parseTensorGetArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, RAI_Tensor *t);

#endif /* SRC_TENSOR_H_ */
