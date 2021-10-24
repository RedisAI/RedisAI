/**
 * tensor.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the RedisAI_TensorType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#pragma once

#include "err.h"
#include "redismodule.h"
#include "redisai.h"
#include "tensor_struct.h"
#include "util/dict.h"
#include "config/config.h"
#include "dlpack/dlpack.h"

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
static const char *RAI_DATATYPE_STR_BOOL = "BOOL";
static const char *RAI_DATATYPE_STR_STRING = "STRING";

#define TENSOR_NONE                0
#define TENSOR_VALUES              (1 << 0)
#define TENSOR_META                (1 << 1)
#define TENSOR_BLOB                (1 << 2)
#define TENSOR_ILLEGAL_VALUES_BLOB (TENSOR_VALUES | TENSOR_BLOB)

/**
 * @brief  Returns the redis module type representing a tensor.
 * @return redis module type representing a tensor.
 */
RedisModuleType *RAI_TensorRedisType(void);

//********************* methods for creating a tensor ***************************
/**
 * Allocate the memory for a new tensor and set RAI_Tensor meta-data based on
 * the passed 'data type`, the specified number of dimensions
 * `ndims`, and n-dimension array `dims`. Note that tensor data is not initialized!
 *
 * @param data_type represents the data type of tensor elements
 * @param dims n-dimensional array ( the dimension values are copied )
 * @param n_dims number of dimensions
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorNew(DLDataType data_type, const size_t *dims, int n_dims);

/**
 * Allocate the memory and set RAI_Tensor meta-data based on
 * the passed 'dataType` string, the specified number of dimensions
 * `ndims`, and n-dimension array `dims`. Note that tensor data is not initialized!
 *
 * @param data_type string containing the data type of tensor elements
 * @param dims n-dimensional array ( the dimension values are copied )
 * @param n_dims number of dimensions
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorCreate(const char *data_type, const long long *dims, int n_dims);

/**
 * Allocate the memory and initialise the RAI_Tensor. Creates a tensor based on
 * the passed data type and with the specified number of dimensions `ndims`,
 * and n-dimension array `dims`. The tensor will be populated with the given values.
 *
 * @param data_type DLDataType that represents the tensor elements data type.
 * @param dims array of size ndims, contains the tensor shapes (the dimension values are copied)
 * @param ndims number of dimensions
 * @param argc number of values to store in the vector (the size of argv array)
 * @param argv array of redis module strings contains the tensor values
 * @param err used to store error status if one occurs
 * @return allocated RAI_Tensor on success, or NULL if operation failed.
 */
RAI_Tensor *RAI_TensorCreateFromValues(DLDataType data_type, const size_t *dims, int n_dims,
                                       int argc, RedisModuleString **argv, RAI_Error *err);

/**
 * Allocate the memory and initialise the RAI_Tensor. Creates a tensor based on
 * the passed data type and with the specified number of dimensions `ndims`,
 * and n-dimension array `dims`. The tensor will be populated with the given data blob.
 *
 * @param data_type DLDataType that represents the tensor elements data type.
 * @param dims array of size ndims, contains the tensor shapes (the dimension values are copied)
 * @param ndims number of dimensions
 * @param tensor_blob a buffer contains the tensor data blob (binary string)
 * @param blob_len number of bytes in tensor blob
 * @param err used to store error status if one occurs
 * @return allocated RAI_Tensor on success, or NULL if operation failed.
 */
RAI_Tensor *RAI_TensorCreateFromBlob(DLDataType data_type, const size_t *dims, int n_dims,
                                     const char *tensor_blob, size_t blob_len, RAI_Error *err);

/**
 * Allocate the memory and initialise the RAI_Tensor, performing a shallow copy
 * of dl_tensor. Beware, this will take ownership of dl_tensor, and only allocate
 * the data for the RAI_Tensor, meaning that the freeing the data of the input
 * dl_tensor, will also free the data of the returned tensor.
 *
 * @param dl_tensor source tensor to shallow copy the data
 * @return allocated RAI_Tensor on success, or NULL if the allocation
 * failed.
 */
RAI_Tensor *RAI_TensorCreateFromDLTensor(DLManagedTensor *dl_tensor);

/**
 * Allocate the memory and initialise an RAI_Tensor which is a concatenation of the input tensors.
 *
 * @param tensors input array of tensors to concatenate
 * @param n number of input tensors
 * @return allocated RAI_Tensor on success, or NULL if the operation failed.
 */
RAI_Tensor *RAI_TensorCreateByConcatenatingTensors(RAI_Tensor **ts, long long n);

/**
 * Allocate the memory and initialise an RAI_Tensor which is a slice of
 * the passed tensor having the given length, starting at the given offset.
 *
 * @param t input tensor
 * @param offset the index of the tensor's first dimension to start copy the data from
 * @param len the length of the sliced tensor (slice ends at offset+len index)
 * @return allocated RAI_Tensor on success, or NULL if the operation failed.
 */
RAI_Tensor *RAI_TensorCreateBySlicingTensor(RAI_Tensor *t, long long offset, long long len);

/**
 * Helper method for creating the DLDataType represented by the input string
 *
 * @param data_type
 * @return the DLDataType represented by the input string
 */
DLDataType RAI_TensorDataTypeFromString(const char *data_type);

//*************** getters and setters ******************************

/**
 * @param tensor
 * @return A pointer to the inner DLTensor field (do not copy).
 */
DLTensor *RAI_TensorGetDLTensor(RAI_Tensor *tensor);

/**
 * Returns the length (i.e., number of elements) of the input tensor
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
 * Get the associated `DLDataType` for the given input tensor
 *
 * @param t input tensor
 * @return the associated `DLDataType` for the given input tensor
 */
DLDataType RAI_TensorDataType(RAI_Tensor *t);

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
 * Returns the size in bytes of the underlying tensor data
 *
 * @param t input tensor
 * @return the size in bytes of the underlying deep learning tensor data
 */
size_t RAI_TensorByteSize(RAI_Tensor *t);

/**
 * Return the pointer to the tensor data blob (do not copy)
 *
 * @param t input tensor
 * @return direct access to the array data pointer
 */
char *RAI_TensorData(RAI_Tensor *t);

/**
 * Return the pointer to the array containing the offset of every string element
 * in the data array
 *
 * @param t input tensor
 * @return direct access to the offsets array pointer
 */
uint64_t *RAI_TensorStringElementsOffsets(RAI_Tensor *tensor);

/**
 * Return the pointer to the array containing the tensors dimensions.
 *
 * @param t input tensor
 * @return direct access to the shpae array pointer
 */
int64_t *RAI_TensorShape(RAI_Tensor *tensor);

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
 * Gets the a const pointer to the string value from the given input tensor, at the given array data
 * pointer position (do not copy the string).
 *
 * @param t tensor to get the data
 * @param i dl_tensor data pointer position
 * @param val value to set the data to
 * @return 1 on success, or 0 if getting the data failed
 */
int RAI_TensorGetValueAsCString(RAI_Tensor *t, long long i, const char **val);

/**
 * sets in data_type_str the string representing the associated DLDataType
 *
 * @param data_type DLDataType
 * @param data_type_str output string to store the associated string representing the
 * DLDataType
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed (unsupported data type)
 */
int RAI_TensorGetDataTypeStr(DLDataType data_type, char *data_type_str);

/**
 * Sets the associated data to the tensor via deep copying the
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

//************************************** tensor memory management
//***********************************
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
 * Frees the memory of the RAI_Tensor when the tensor reference count reaches 0.
 * It is safe to call this function with a NULL input tensor.
 *
 * @param t tensor
 */
void RAI_TensorFree(RAI_Tensor *t);

//*************** methods for retrieval and replicating tensor from keyspace ***************
/**
 * Helper method to open a key handler for the tensor data type
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param mode key access mode
 * @param err used to store error status if one occurs
 * @return REDISMODULE_OK if it's possible to store at the specified key handle
 * the tensor type, or REDISMODULE_ERR if is the key not associated with a
 * tensor type.
 */
int RAI_TensorOpenKey(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
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
 * @param err used to store error status if one occurs.
 * @return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable, or REDISMODULE_ERR if there was
 * an error getting the Tensor
 */
int RAI_TensorGetFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Tensor **tensor, int mode, RAI_Error *err);

/**
 * Helper method to replicate a tensor via an AI.TENSORSET command to the
 * replicas. This is used on MODELRUN, SCRIPTRUN, DAGRUN as a way to ensure that
 * the results present on replicas match the results present on master
 *
 * @param ctx Context in which Redis modules operate
 * @param key Destination key name
 * @param t source tensor
 */
void RAI_TensorReplicate(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t);

/**
 * Helper method to return a tensor to the client in a response to AI.TENSORGET
 *
 * @param ctx Context in which Redis modules operate.
 * @param fmt The format in which tensor is returned (META and/or VALUES/BLOB).
 * @param t The tensor to reply with.

 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise.
 */
int RAI_TensorReply(RedisModuleCtx *ctx, uint fmt, RAI_Tensor *t);
