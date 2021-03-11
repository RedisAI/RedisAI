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
#include "redisai.h"
#include "tensor_struct.h"
#include "util/dict.h"

#define TENSORALLOC_NONE   0
#define TENSORALLOC_ALLOC  1
#define TENSORALLOC_CALLOC 2

// Numeric data type of tensor elements, one of FLOAT, DOUBLE, INT8, INT16,
// INT32, INT64, UINT8, UINT16
static const char *RAI_DATATYPE_STR_FLOAT = "FLOAT";
static const char *RAI_DATATYPE_STR_DOUBLE = "DOUBLE";
static const char *RAI_DATATYPE_STR_INT8 = "INT8";
static const char *RAI_DATATYPE_STR_INT16 = "INT16";
static const char *RAI_DATATYPE_STR_INT32 = "INT32";
static const char *RAI_DATATYPE_STR_INT64 = "INT64";
static const char *RAI_DATATYPE_STR_UINT8 = "UINT8";
static const char *RAI_DATATYPE_STR_UINT16 = "UINT16";

#define TENSOR_NONE                0
#define TENSOR_VALUES              (1 << 0)
#define TENSOR_META                (1 << 1)
#define TENSOR_BLOB                (1 << 2)
#define TENSOR_ILLEGAL_VALUES_BLOB (TENSOR_VALUES | TENSOR_BLOB)

extern RedisModuleType *RedisAI_TensorType;

/**
 * Helper method to register the tensor type exported by the module.
 *
 * @param ctx Context in which Redis modules operate
 * @return
 */
int RAI_TensorInit(RedisModuleCtx *ctx);

/**
 * @brief Allocate an empty tensor with no data.
 * @note The new tensor ref coutn is 1.
 *
 * @return RAI_Tensor* - a pointer to the new tensor.
 */
RAI_Tensor *RAI_TensorNew(void);

/**
 * Allocate the memory and initialise the RAI_Tensor. Creates a tensor based on
 * the passed 'dataType` string and with the specified number of dimensions
 * `ndims`, and n-dimension array `dims`.
 *
 * @param dataType string containing the numeric data type of tensor elements
 * @param dims n-dimensional array ( the dimension values are copied )
 * @param ndims number of dimensions
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorCreate(const char *dataType, long long *dims, int ndims);

/**
 * Allocate the memory and initialise the RAI_Tensor. Creates a tensor based on
 * the passed 'DLDataType` and with the specified number of dimensions `ndims`,
 * and n-dimension array `dims`. Depending on the passed `tensorAllocMode`, the
 * DLTensor data will be either allocated or no allocation is performed thus
 * enabling shallow copy of data (no alloc)
 *
 * @param dtype DLDataType
 * @param dims n-dimensional array ( the dimension values are copied )
 * @param ndims number of dimensions
 * @param empty True if creating an empty tensor (need to be initialized)
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorCreateWithDLDataType(DLDataType dtype, long long *dims, int ndims,
                                           bool empty);

/**
 * Allocate the memory for a new Tensor and copy data fom a tensor to it.
 *
 * @param t Source tensor to copy.
 * @param result Destination tensor to copy.
 * @return 0 on success, or 1 if the copy failed
 * failed.
 */
int RAI_TensorDeepCopy(RAI_Tensor *t, RAI_Tensor **dest);

/**
 * Allocate the memory and initialise the RAI_Tensor, performing a shallow copy
 * of dl_tensor. Beware, this will take ownership of dltensor, and only allocate
 * the data for the RAI_Tensor, meaning that the freeing the data of the input
 * dl_tensor, will also free the data of the returned tensor.
 *
 * @param dl_tensor source tensor to shallow copy the data
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorCreateFromDLTensor(DLManagedTensor *dl_tensor);

/**
 * Allocate the memory and initialise the RAI_Tensor, performing a deep copy of
 * the passed array of tensors.
 *
 * @param ts input array of tensors
 * @param n number of input tensors
 * @return allocated RAI_Tensor on success, or NULL if the allocation and deep
 * copy failed failed.
 */
RAI_Tensor *RAI_TensorCreateByConcatenatingTensors(RAI_Tensor **ts, long long n);

/**
 * Allocate the memory and initialise the RAI_Tensor, performing a deep copy of
 * the passed tensor, at the given data offset and length.
 *
 * @param t input tensor
 * @param offset
 * @param len
 * @return allocated RAI_Tensor on success, or NULL if the allocation and deep
 * copy failed failed.
 */
RAI_Tensor *RAI_TensorCreateBySlicingTensor(RAI_Tensor *t, long long offset, long long len);

/**
 * Returns the length of the input tensor
 *
 * @param t input tensor
 * @return the length of the input tensor
 */
size_t RAI_TensorLength(RAI_Tensor *t);

/**
 * Returns the size in bytes of each element of the tensor
 *
 * @param t input tensor
 * @return size in bytes of each the underlying tensor data type
 */
size_t RAI_TensorDataSize(RAI_Tensor *t);

/**
 * Returns the size in bytes of the given DLDataType
 *
 * @param dtype DLDataType
 * @return size in bytes of each the given DLDataType
 */
size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype);

/**
 * Returns the size in bytes of the associated DLDataType represented by the
 * input string
 *
 * @param dataType
 * @return size in bytes of each the underlying tensor data type
 */
size_t RAI_TensorDataSizeFromString(const char *dataType);

/**
 * Get the associated `DLDataType` for the given input tensor
 *
 * @param t input tensor
 * @return the associated `DLDataType` for the given input tensor
 */
DLDataType RAI_TensorDataType(RAI_Tensor *t);

/**
 * Check whether two tensors have the same data type
 *
 * @param t1 input tensor
 * @param t2 input tensor
 * @return 1 if data types match, 0 otherwise
 */
int RAI_TensorIsDataTypeEqual(RAI_Tensor *t1, RAI_Tensor *t2);

/**
 * Returns the DLDataType represented by the input string
 *
 * @param dataType
 * @return the DLDataType represented by the input string
 */
DLDataType RAI_TensorDataTypeFromString(const char *dataType);

/**
 * sets in dtypestr the string representing the associated DLDataType
 *
 * @param dtype DLDataType
 * @param dtypestr output string to store the associated string representing the
 * DLDataType
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed
 */
int Tensor_DataTypeStr(DLDataType dtype, char *dtypestr);

/**
 * Frees the memory of the RAI_Tensor when the tensor reference count reaches 0.
 * It is safe to call this function with a NULL input tensor.
 *
 * @param t tensor
 */
void RAI_TensorFree(RAI_Tensor *t);

/**
 * Sets the associated data to the deep learning tensor via deep copying the
 * passed data.
 *
 * @param t tensor to set the data
 * @param data input data
 * @param len input data length
 * @return 1 on success
 */
int RAI_TensorSetData(RAI_Tensor *t, const char *data, size_t len);

/**
 * Sets the long value for the given tensor, at the given array data pointer
 * position
 *
 * @param t tensor to set the data
 * @param i dl_tensor data pointer position
 * @param val value to set the data from
 * @return 0 on success, or 1 if the setting failed
 */
int RAI_TensorSetValueFromLongLong(RAI_Tensor *t, long long i, long long val);

/**
 * Sets the double value for the given tensor, at the given array data pointer
 * position
 *
 * @param t tensor to set the data
 * @param i dl_tensor data pointer position
 * @param val value to set the data from
 * @return 1 on success, or 0 if the setting failed
 */
int RAI_TensorSetValueFromDouble(RAI_Tensor *t, long long i, double val);

/**
 * Gets the double value from the given input tensor, at the given array data
 * pointer position
 *
 * @param t tensor to get the data
 * @param i dl_tensor data pointer position
 * @param val value to set the data to
 * @return 1 on success, or 0 if getting the data failed
 */
int RAI_TensorGetValueAsDouble(RAI_Tensor *t, long long i, double *val);

/**
 * Gets the long value from the given input tensor, at the given array data
 * pointer position
 *
 * @param t tensor to get the data
 * @param i dl_tensor data pointer position
 * @param val value to set the data to
 * @return 1 on success, or 0 if getting the data failed
 */
int RAI_TensorGetValueAsLongLong(RAI_Tensor *t, long long i, long long *val);

/**
 * Every call to this function, will make the RAI_Tensor 't' requiring an
 * additional call to RAI_TensorFree() in order to really free the tensor.
 * Returns a shallow copy of the tensor.
 *
 * @param t input tensor
 * @return shallow copy of the tensor
 */
RAI_Tensor *RAI_TensorGetShallowCopy(RAI_Tensor *t);

/**
 * Returns the number of dimensions for the given input tensor
 *
 * @param t input tensor
 * @return number of dimensions for the given input tensor
 */
int RAI_TensorNumDims(RAI_Tensor *t);

/**
 * Returns the dimension length for the given input tensor and dimension
 *
 * @param t input tensor
 * @param dim dimension
 * @return the dimension length
 */
long long RAI_TensorDim(RAI_Tensor *t, int dim);

/**
 * Returns the size in bytes of the underlying deep learning tensor data
 *
 * @param t input tensor
 * @return the size in bytes of the underlying deep learning tensor data
 */
size_t RAI_TensorByteSize(RAI_Tensor *t);

/**
 * Return the pointer the the deep learning tensor data
 *
 * @param t input tensor
 * @return direct access to the array data pointer
 */
char *RAI_TensorData(RAI_Tensor *t);

/**
 * Helper method to open a key handler for the tensor data type
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param mode key access mode
 * @return REDISMODULE_OK if it's possible to store at the specified key handle
 * the tensor type, or REDISMODULE_ERR if is the key not associated with a
 * tensor type.
 */
int RAI_OpenKey_Tensor(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                       int mode, RAI_Error *err);

/**
 * Helper method to get Tensor from keyspace. In case of a failure an
 * error is documented.
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param tensor destination tensor structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable, or REDISMODULE_ERR if there was
 * an error getting the Tensor
 */
int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Tensor **tensor, int mode, RAI_Error *err);

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
void RedisAI_ReplicateTensorSet(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t);

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param t Destination tensor to store the parsed data
 * @param enforceArity flag whether to enforce arity checking
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RAI_parseTensorSetArgs(RedisModuleString **argv, int argc, RAI_Tensor **t, int enforceArity,
                           RAI_Error *error);

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return The format in which tensor is returned.
 */

uint ParseTensorGetArgs(RAI_Error *error, RedisModuleString **argv, int argc);

/**
 * Helper method to return a tensor to the client in a response to AI.TENSORGET
 *
 * @param ctx Context in which Redis modules operate.
 * @param fmt The format in which tensor is returned.
 * @param t The tensor to reply with.

 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise.
 */

int ReplyWithTensor(RedisModuleCtx *ctx, uint fmt, RAI_Tensor *t);

/**
 * @brief  Returns the redis module type representing a tensor.
 * @return redis module type representing a tensor.
 */
RedisModuleType *RAI_TensorRedisType(void);

#endif /* SRC_TENSOR_H_ */
