/**
 * tensor.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the RedisAI_TensorType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include "config.h"
#include "dlpack/dlpack.h"
#include "err.h"
#include "redismodule.h"
#include "tensor_struct.h"
#include "util/dict.h"

#define TENSORALLOC_NONE 0
#define TENSORALLOC_ALLOC 1
#define TENSORALLOC_CALLOC 2

// Numeric data type of tensor elements, one of FLOAT, DOUBLE, INT8, INT16,
// INT32, INT64, UINT8, UINT16
static const char* RAI_DATATYPE_STR_FLOAT = "FLOAT";
static const char* RAI_DATATYPE_STR_DOUBLE = "DOUBLE";
static const char* RAI_DATATYPE_STR_INT8 = "INT8";
static const char* RAI_DATATYPE_STR_INT16 = "INT16";
static const char* RAI_DATATYPE_STR_INT32 = "INT32";
static const char* RAI_DATATYPE_STR_INT64 = "INT64";
static const char* RAI_DATATYPE_STR_UINT8 = "UINT8";
static const char* RAI_DATATYPE_STR_UINT16 = "UINT16";

extern RedisModuleType* RedisAI_TensorType;

int RAI_TensorInit(RedisModuleCtx* ctx);
RAI_Tensor* RAI_TensorCreate(const char* dataType, long long* dims, int ndims,
                             int hasdata);
RAI_Tensor* RAI_TensorCreateWithDLDataType(DLDataType dtype, long long* dims,
                                           int ndims, int tensorAllocMode);

/**
 * Allocate the memory for a new Tensor and copy data fom a tensor to it.
 * @param t Source tensor to copy.
 * @param result Destination tensor to copy.
 * @return 0 on success, or 1 if the copy failed
 * failed.
 */
int RAI_TensorCopyTensor(RAI_Tensor* t, RAI_Tensor** dest);
RAI_Tensor* RAI_TensorCreateFromDLTensor(DLManagedTensor* dl_tensor);
RAI_Tensor* RAI_TensorCreateByConcatenatingTensors(RAI_Tensor** ts,
                                                   long long n);
RAI_Tensor* RAI_TensorCreateBySlicingTensor(RAI_Tensor* t, long long offset,
                                            long long len);
size_t RAI_TensorLength(RAI_Tensor* t);
size_t RAI_TensorDataSize(RAI_Tensor* t);
size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype);
size_t RAI_TensorDataSizeFromString(const char* dataType);
DLDataType RAI_TensorDataType(RAI_Tensor* t);
DLDataType RAI_TensorDataTypeFromString(const char* dataType);
int Tensor_DataTypeStr(DLDataType dtype, char** dtypestr);
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

/**
 * Helper method to open a key handler for the tensor data type
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On sucess it contains an handle representing
 * a Redis key with the requested access mode
 * @param mode key access mode
 * @return REDISMODULE_OK if it's possible to store at the specified key handle
 * the tensor type, or REDISMODULE_ERR if is the key not associated with a
 * tensor type.
 */
int RAI_OpenKey_Tensor(RedisModuleCtx* ctx, RedisModuleString* keyName,
                       RedisModuleKey** key, int mode);

/**
 * Helper method to get Tensor from keyspace. In the case of failure the key is
 * closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On sucess it contains an handle representing
 * a Redis key with the requested access mode
 * @param tensor destination tensor structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable, or REDISMODULE_ERR if there was
 * an error getting the Tensor
 */
int RAI_GetTensorFromKeyspace(RedisModuleCtx* ctx, RedisModuleString* keyName,
                              RedisModuleKey** key, RAI_Tensor** tensor,
                              int mode);

/**
 * Helper method to get Tensor from local context ( no keyspace access )
 *
 * @param ctx Context in which Redis modules operate
 * @param localContextDict local non-blocking hash table containing DAG's
 * tensors
 * @param localContextKey key name
 * @param tensor destination tensor
 * @param error error data structure to store error message in the case of
 * failure
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed
 */
int RAI_getTensorFromLocalContext(RedisModuleCtx* ctx,
                                  AI_dict* localContextDict,
                                  const char* localContextKey,
                                  RAI_Tensor** tensor, RAI_Error* error);

/**
 * Helper method to replicate a tensor via an AI.TENSORSET command to the
 * replicas. This is used on MODELRUN, SCRIPTRUN, DAGRUN as a way to ensure that
 * the results present on replicas match the results present on master ( since
 * multiple modelruns/scripts are not ensured to have the same output values
 * (non-deterministic) ).
 *
 * @param ctx Context in which Redis modules operate
 * @param key Destination key name
 * @param t source tensor
 */
void RedisAI_ReplicateTensorSet(RedisModuleCtx* ctx, RedisModuleString* key,
                                RAI_Tensor* t);

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param t Destination tensor to store the parsed data
 * @param enforceArity flag wether to enforce arity checking
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RAI_parseTensorSetArgs(RedisModuleCtx* ctx, RedisModuleString** argv,
                           int argc, RAI_Tensor** t, int enforceArity,
                           RAI_Error* error);

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param t Destination tensor to store the parsed data
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RAI_parseTensorGetArgs(RedisModuleCtx* ctx, RedisModuleString** argv,
                           int argc, RAI_Tensor* t);

#endif /* SRC_TENSOR_H_ */
