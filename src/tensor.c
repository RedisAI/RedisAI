/**
 * tensor.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisAI_TensorType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include "tensor.h"
#include "err.h"
#include "arr.h"
#include "redisai.h"
#include "version.h"
#include "rmutil/alloc.h"
#include "tensor_struct.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <strings.h>

DLDataType RAI_TensorDataTypeFromString(const char *typestr) {
    if (strcasecmp(typestr, RAI_DATATYPE_STR_FLOAT) == 0) {
        return (DLDataType){.code = kDLFloat, .bits = 32, .lanes = 1};
    }
    if (strcasecmp(typestr, RAI_DATATYPE_STR_DOUBLE) == 0) {
        return (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1};
    }
    if (strncasecmp(typestr, "INT", 3) == 0) {
        const char *bitstr = typestr + 3;
        if (strcmp(bitstr, "8") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 8, .lanes = 1};
        }
        if (strcmp(bitstr, "16") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 16, .lanes = 1};
        }
        if (strcmp(bitstr, "32") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
        }
        if (strcmp(bitstr, "64") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
        }
    }
    if (strncasecmp(typestr, "UINT", 4) == 0) {
        const char *bitstr = typestr + 4;
        if (strcmp(bitstr, "8") == 0) {
            return (DLDataType){.code = kDLUInt, .bits = 8, .lanes = 1};
        }
        if (strcmp(bitstr, "16") == 0) {
            return (DLDataType){.code = kDLUInt, .bits = 16, .lanes = 1};
        }
    }
    return (DLDataType){.bits = 0};
}

static size_t Tensor_DataTypeSize(DLDataType dtype) { return dtype.bits / 8; }

int Tensor_DataTypeStr(DLDataType dtype, char *dtypestr) {
    int result = REDISMODULE_ERR;

    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) {
            strcpy(dtypestr, RAI_DATATYPE_STR_FLOAT);
            result = REDISMODULE_OK;
        } else if (dtype.bits == 64) {
            strcpy(dtypestr, RAI_DATATYPE_STR_DOUBLE);
            result = REDISMODULE_OK;
        }
    } else if (dtype.code == kDLInt) {
        if (dtype.bits == 8) {
            strcpy(dtypestr, RAI_DATATYPE_STR_INT8);
            result = REDISMODULE_OK;
        } else if (dtype.bits == 16) {
            strcpy(dtypestr, RAI_DATATYPE_STR_INT16);
            result = REDISMODULE_OK;
        } else if (dtype.bits == 32) {
            strcpy(dtypestr, RAI_DATATYPE_STR_INT32);
            result = REDISMODULE_OK;
        } else if (dtype.bits == 64) {
            strcpy(dtypestr, RAI_DATATYPE_STR_INT64);
            result = REDISMODULE_OK;
        }
    } else if (dtype.code == kDLUInt) {
        if (dtype.bits == 8) {
            strcpy(dtypestr, RAI_DATATYPE_STR_UINT8);
            result = REDISMODULE_OK;
        } else if (dtype.bits == 16) {
            strcpy(dtypestr, RAI_DATATYPE_STR_UINT16);
            result = REDISMODULE_OK;
        }
    }
    return result;
}

RAI_Tensor *RAI_TensorNew(void) {
    RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->refCount = 1;
    ret->len = LEN_UNKOWN;
}

RAI_Tensor *RAI_TensorCreateWithDLDataType(DLDataType dtype, long long *dims, int ndims,
                                           int tensorAllocMode) {

    size_t dtypeSize = Tensor_DataTypeSize(dtype);
    if (dtypeSize == 0) {
        return NULL;
    }

    RAI_Tensor *ret = RAI_TensorNew();
    int64_t *shape = RedisModule_Alloc(ndims * sizeof(*shape));
    int64_t *strides = RedisModule_Alloc(ndims * sizeof(*strides));

    size_t len = 1;
    for (int64_t i = 0; i < ndims; ++i) {
        shape[i] = dims[i];
        strides[i] = 1;
        len *= dims[i];
    }
    for (int64_t i = ndims - 2; i >= 0; --i) {
        strides[i] *= strides[i + 1] * shape[i + 1];
    }

    DLContext ctx = (DLContext){.device_type = kDLCPU, .device_id = 0};
    void *data = NULL;
    switch (tensorAllocMode) {
    case TENSORALLOC_ALLOC:
        data = RedisModule_Alloc(len * dtypeSize);
        break;
    case TENSORALLOC_CALLOC:
        data = RedisModule_Calloc(len, dtypeSize);
        break;
    case TENSORALLOC_NONE:
        /* shallow copy no alloc */
    default:
        /* assume TENSORALLOC_NONE
        shallow copy no alloc */
        break;
    }

    ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.ctx = ctx,
                                                            .data = data,
                                                            .ndim = ndims,
                                                            .dtype = dtype,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = 0},
                                    .manager_ctx = NULL,
                                    .deleter = NULL};

    return ret;
}

void RAI_RStringDataTensorDeleter(DLManagedTensor *arg) {
    if (arg->dl_tensor.shape) {
        RedisModule_Free(arg->dl_tensor.shape);
    }
    if (arg->dl_tensor.strides) {
        RedisModule_Free(arg->dl_tensor.strides);
    }
    if (arg->manager_ctx) {
        RedisModuleString *rstr = (RedisModuleString *)arg->manager_ctx;
        RedisModule_FreeString(NULL, rstr);
    }
    RedisModule_Free(arg->dl_tensor.data);
    RedisModule_Free(arg);
}

RAI_Tensor *_TensorCreateWithDLDataTypeAndRString(DLDataType dtype, size_t dtypeSize,
                                                  long long *dims, int ndims,
                                                  RedisModuleString *rstr, RAI_Error *err) {

    int64_t *shape = RedisModule_Alloc(ndims * sizeof(*shape));
    int64_t *strides = RedisModule_Alloc(ndims * sizeof(*strides));

    size_t len = 1;
    for (int64_t i = 0; i < ndims; ++i) {
        shape[i] = dims[i];
        strides[i] = 1;
        len *= dims[i];
    }
    for (int64_t i = ndims - 2; i >= 0; --i) {
        strides[i] *= strides[i + 1] * shape[i + 1];
    }

    DLContext ctx = (DLContext){.device_type = kDLCPU, .device_id = 0};
    size_t nbytes = len * dtypeSize;

    size_t blob_len;
    const char *blob = RedisModule_StringPtrLen(rstr, &blob_len);
    if (blob_len != nbytes) {
        RedisModule_Free(shape);
        RedisModule_Free(strides);
        RAI_SetError(err, RAI_ETENSORSET, "ERR data length does not match tensor shape and type");
        return NULL;
    }
    char *data = RedisModule_Alloc(nbytes);
    memcpy(data, blob, nbytes);
    RAI_HoldString(NULL, rstr);

    RAI_Tensor *ret = RAI_TensorNew();
    ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.ctx = ctx,
                                                            .data = data,
                                                            .ndim = ndims,
                                                            .dtype = dtype,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = 0},
                                    .manager_ctx = rstr,
                                    .deleter = RAI_RStringDataTensorDeleter};

    return ret;
}

RAI_Tensor *RAI_TensorCreate(const char *dataType, long long *dims, int ndims, int hasdata) {
    DLDataType dtype = RAI_TensorDataTypeFromString(dataType);
    return RAI_TensorCreateWithDLDataType(dtype, dims, ndims, TENSORALLOC_ALLOC);
}

#if 0
void RAI_TensorMoveFrom(RAI_Tensor* dst, RAI_Tensor* src) {
  if (--dst->refCount <= 0){
    RedisModule_Free(t->tensor.shape);
    if (t->tensor.strides) {
      RedisModule_Free(t->tensor.strides);
    }
    RedisModule_Free(t->tensor.data);
    RedisModule_Free(t);
  }
  dst->tensor.ctx = src->tensor.ctx;
  dst->tensor.data = src->tensor.data;

  dst->refCount = 1;
}
#endif

RAI_Tensor *RAI_TensorCreateByConcatenatingTensors(RAI_Tensor **ts, long long n) {

    if (n == 0) {
        return NULL;
    }

    long long total_batch_size = 0;
    long long batch_sizes[n];
    long long batch_offsets[n];

    const long long ndims = RAI_TensorNumDims(ts[0]);
    long long dims[ndims];

    // TODO check that all tensors have compatible dims

    for (long long i = 0; i < n; i++) {
        batch_sizes[i] = RAI_TensorDim(ts[i], 0);
        total_batch_size += batch_sizes[i];
    }

    batch_offsets[0] = 0;
    for (long long i = 1; i < n; i++) {
        batch_offsets[i] = batch_offsets[i - 1] + batch_sizes[i - 1];
    }

    long long sample_size = 1;

    for (long long i = 1; i < ndims; i++) {
        dims[i] = RAI_TensorDim(ts[0], i);
        sample_size *= dims[i];
    }
    dims[0] = total_batch_size;

    const long long dtype_size = RAI_TensorDataSize(ts[0]);

    DLDataType dtype = RAI_TensorDataType(ts[0]);

    RAI_Tensor *ret = RAI_TensorCreateWithDLDataType(dtype, dims, ndims, TENSORALLOC_ALLOC);

    for (long long i = 0; i < n; i++) {
        memcpy(RAI_TensorData(ret) + batch_offsets[i] * sample_size * dtype_size,
               RAI_TensorData(ts[i]), RAI_TensorByteSize(ts[i]));
    }

    return ret;
}

RAI_Tensor *RAI_TensorCreateBySlicingTensor(RAI_Tensor *t, long long offset, long long len) {

    const long long ndims = RAI_TensorNumDims(t);
    long long dims[ndims];

    const long long dtype_size = RAI_TensorDataSize(t);
    long long sample_size = 1;

    for (long long i = 1; i < ndims; i++) {
        dims[i] = RAI_TensorDim(t, i);
        sample_size *= dims[i];
    }

    dims[0] = len;

    DLDataType dtype = RAI_TensorDataType(t);

    RAI_Tensor *ret = RAI_TensorCreateWithDLDataType(dtype, dims, ndims, TENSORALLOC_ALLOC);

    memcpy(RAI_TensorData(ret), RAI_TensorData(t) + offset * sample_size * dtype_size,
           len * sample_size * dtype_size);

    return ret;
}

/**
 * Allocate the memory for a new Tensor and copy data fom a tensor to it.
 * @param t Source tensor to copy.
 * @param result Destination tensor to copy.
 * @return 0 on success, or 1 if the copy failed
 * failed.
 */
int RAI_TensorDeepCopy(RAI_Tensor *t, RAI_Tensor **dest) {
    const long long ndims = RAI_TensorNumDims(t);
    long long dims[ndims];

    const long long dtype_size = RAI_TensorDataSize(t);
    long long sample_size = 1;

    for (size_t i = 0; i < ndims; i++) {
        dims[i] = RAI_TensorDim(t, i);
        sample_size *= dims[i];
    }

    DLDataType dtype = RAI_TensorDataType(t);

    RAI_Tensor *ret = RAI_TensorCreateWithDLDataType(dtype, dims, ndims, TENSORALLOC_ALLOC);

    memcpy(RAI_TensorData(ret), RAI_TensorData(t), sample_size * dtype_size);
    *dest = ret;
    return 0;
}

// Beware: this will take ownership of dltensor
RAI_Tensor *RAI_TensorCreateFromDLTensor(DLManagedTensor *dl_tensor) {

    RAI_Tensor *ret = RAI_TensorNew();

    ret->tensor =
        (DLManagedTensor){.dl_tensor = (DLTensor){.ctx = dl_tensor->dl_tensor.ctx,
                                                  .data = dl_tensor->dl_tensor.data,
                                                  .ndim = dl_tensor->dl_tensor.ndim,
                                                  .dtype = dl_tensor->dl_tensor.dtype,
                                                  .shape = dl_tensor->dl_tensor.shape,
                                                  .strides = dl_tensor->dl_tensor.strides,
                                                  .byte_offset = dl_tensor->dl_tensor.byte_offset},
                          .manager_ctx = dl_tensor->manager_ctx,
                          .deleter = dl_tensor->deleter};

    return ret;
}

DLDataType RAI_TensorDataType(RAI_Tensor *t) { return t->tensor.dl_tensor.dtype; }

int RAI_TensorIsDataTypeEqual(RAI_Tensor *t1, RAI_Tensor *t2) {
    return t1->tensor.dl_tensor.dtype.bits == t2->tensor.dl_tensor.dtype.bits &&
           t1->tensor.dl_tensor.dtype.code == t2->tensor.dl_tensor.dtype.code &&
           t1->tensor.dl_tensor.dtype.lanes == t2->tensor.dl_tensor.dtype.lanes;
}

size_t RAI_TensorLength(RAI_Tensor *t) {
    if (t->len == LEN_UNKOWN) {
        int64_t *shape = t->tensor.dl_tensor.shape;
        size_t len = 1;
        for (size_t i = 0; i < t->tensor.dl_tensor.ndim; ++i) {
            len *= shape[i];
        }
        t->len = len;
    }
    return t->len;
}

size_t RAI_TensorDataSize(RAI_Tensor *t) { return Tensor_DataTypeSize(RAI_TensorDataType(t)); }

size_t RAI_TensorDataSizeFromString(const char *dataTypeStr) {
    DLDataType dtype = RAI_TensorDataTypeFromString(dataTypeStr);
    return Tensor_DataTypeSize(dtype);
}

size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype) { return Tensor_DataTypeSize(dtype); }

void RAI_TensorFree(RAI_Tensor *t) {
    if (t) {
        if (__atomic_sub_fetch(&t->refCount, 1, __ATOMIC_RELAXED) <= 0) {
            if (t->tensor.deleter) {
                t->tensor.deleter(&t->tensor);
            } else {
                if (t->tensor.dl_tensor.shape) {
                    RedisModule_Free(t->tensor.dl_tensor.shape);
                }
                if (t->tensor.dl_tensor.strides) {
                    RedisModule_Free(t->tensor.dl_tensor.strides);
                }
                if (t->tensor.dl_tensor.data) {
                    RedisModule_Free(t->tensor.dl_tensor.data);
                }
                RedisModule_Free(t);
            }
        }
    }
}

int RAI_TensorSetData(RAI_Tensor *t, const char *data, size_t len) {
    memcpy(t->tensor.dl_tensor.data, data, len);
    return 1;
}

int RAI_TensorSetValueFromLongLong(RAI_Tensor *t, long long i, long long val) {
    DLDataType dtype = t->tensor.dl_tensor.dtype;
    void *data = t->tensor.dl_tensor.data;

    if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        case 8:
            ((int8_t *)data)[i] = val;
            break;
            break;
        case 16:
            ((int16_t *)data)[i] = val;
            break;
            break;
        case 32:
            ((int32_t *)data)[i] = val;
            break;
            break;
        case 64:
            ((int64_t *)data)[i] = val;
            break;
            break;
        default:
            return 0;
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            ((uint8_t *)data)[i] = val;
            break;
            break;
        case 16:
            ((uint16_t *)data)[i] = val;
            break;
            break;
        case 32:
            ((uint32_t *)data)[i] = val;
            break;
            break;
        case 64:
            ((uint64_t *)data)[i] = val;
            break;
            break;
        default:
            return 0;
        }
    } else {
        return 0;
    }
    return 1;
}

int RAI_TensorSetValueFromDouble(RAI_Tensor *t, long long i, double val) {
    DLDataType dtype = t->tensor.dl_tensor.dtype;
    void *data = t->tensor.dl_tensor.data;

    if (dtype.code == kDLFloat) {
        switch (dtype.bits) {
        case 32:
            ((float *)data)[i] = val;
            break;
        case 64:
            ((double *)data)[i] = val;
            break;
        default:
            return 0;
        }
    } else {
        return 0;
    }
    return 1;
}

int RAI_TensorGetValueAsDouble(RAI_Tensor *t, long long i, double *val) {
    DLDataType dtype = t->tensor.dl_tensor.dtype;
    void *data = t->tensor.dl_tensor.data;

    // TODO: check i is in bound
    if (dtype.code == kDLFloat) {
        switch (dtype.bits) {
        case 32:
            *val = ((float *)data)[i];
            break;
        case 64:
            *val = ((double *)data)[i];
            break;
        default:
            return REDISMODULE_ERR;
        }
    } else {
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int RAI_TensorGetValueAsLongLong(RAI_Tensor *t, long long i, long long *val) {
    DLDataType dtype = t->tensor.dl_tensor.dtype;
    void *data = t->tensor.dl_tensor.data;

    // TODO: check i is in bound

    if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        case 8:
            *val = ((int8_t *)data)[i];
            break;
        case 16:
            *val = ((int16_t *)data)[i];
            break;
        case 32:
            *val = ((int32_t *)data)[i];
            break;
        case 64:
            *val = ((int64_t *)data)[i];
            break;
        default:
            return REDISMODULE_ERR;
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            *val = ((uint8_t *)data)[i];
            break;
        case 16:
            *val = ((uint16_t *)data)[i];
            break;
        case 32:
            *val = ((uint32_t *)data)[i];
            break;
        case 64:
            *val = ((uint64_t *)data)[i];
            break;
        default:
            return REDISMODULE_ERR;
        }
    } else {
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

RAI_Tensor *RAI_TensorGetShallowCopy(RAI_Tensor *t) {
    __atomic_fetch_add(&t->refCount, 1, __ATOMIC_RELAXED);
    return t;
}

int RAI_TensorNumDims(RAI_Tensor *t) { return t->tensor.dl_tensor.ndim; }

long long RAI_TensorDim(RAI_Tensor *t, int i) { return t->tensor.dl_tensor.shape[i]; }

size_t RAI_TensorByteSize(RAI_Tensor *t) {
    // TODO: as per dlpack it should be
    //   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    return Tensor_DataTypeSize(RAI_TensorDataType(t)) * RAI_TensorLength(t);
}

char *RAI_TensorData(RAI_Tensor *t) { return t->tensor.dl_tensor.data; }

int RAI_OpenKey_Tensor(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                       int mode, RAI_Error *err) {
    *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        return REDISMODULE_OK;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RedisModule_CloseKey(*key);
        RAI_SetError(err, RAI_ETENSORSET, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Tensor **tensor, int mode, RAI_Error *err) {
    *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(*key);
        RAI_SetError(err, RAI_ETENSORGET, "ERR tensor key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RedisModule_CloseKey(*key);
        RAI_SetError(err, RAI_ETENSORGET, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *tensor = RedisModule_ModuleTypeGetValue(*key);
    RedisModule_CloseKey(*key);
    return REDISMODULE_OK;
}

void RedisAI_ReplicateTensorSet(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t) {
    long long ndims = RAI_TensorNumDims(t);

    char dtypestr[8];
    const int status = Tensor_DataTypeStr(RAI_TensorDataType(t), dtypestr);
    RedisModule_Assert(status == REDISMODULE_OK);

    char *data = RAI_TensorData(t);
    long long size = RAI_TensorByteSize(t);

    RedisModuleString *dims[ndims];

    for (long long i = 0; i < ndims; i++) {
        dims[i] = RedisModule_CreateStringFromLongLong(ctx, RAI_TensorDim(t, i));
    }

    RedisModule_Replicate(ctx, "AI.TENSORSET", "scvcb", key, dtypestr, dims, ndims, "BLOB", data,
                          size);

    for (long long i = 0; i < ndims; i++) {
        RedisModule_FreeString(ctx, dims[i]);
    }
}

int RAI_parseTensorSetArgs(RedisModuleString **argv, int argc, RAI_Tensor **t, int enforceArity,
                           RAI_Error *error) {
    if (argc < 4) {
        RAI_SetError(error, RAI_ETENSORSET, "wrong number of arguments for 'AI.TENSORSET' command");
        return -1;
    }

    // get the tensor datatype
    const char *typestr = RedisModule_StringPtrLen(argv[2], NULL);
    DLDataType datatype = RAI_TensorDataTypeFromString(typestr);
    size_t datasize = Tensor_DataTypeSize(datatype);
    if (datasize == 0) {
        RAI_SetError(error, RAI_ETENSORSET, "ERR invalid data type");
        return -1;
    }

    const char *fmtstr;
    int datafmt = TENSOR_NONE;
    int tensorAllocMode = TENSORALLOC_CALLOC;
    size_t ndims = 0;
    long long len = 1;
    long long *dims = (long long *)array_new(long long, 1);
    size_t argpos = 3;
    long long remaining_args = argc - 1;
    size_t expected_nvalues = 0;
    size_t current_nvalues = 0;

    for (; argpos <= argc - 1; argpos++) {
        const char *opt = RedisModule_StringPtrLen(argv[argpos], NULL);
        remaining_args = argc - 1 - argpos;
        if (!strcasecmp(opt, "BLOB")) {
            datafmt = TENSOR_BLOB;
            tensorAllocMode = TENSORALLOC_CALLOC;
            // if we've found the dataformat there are no more dimensions
            // check right away if the arity is correct
            if (remaining_args != 1 && enforceArity == 1) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR wrong number of arguments for 'AI.TENSORSET' command");
                return -1;
            }
            argpos++;
            break;
        } else if (!strcasecmp(opt, "VALUES")) {
            datafmt = TENSOR_VALUES;
            tensorAllocMode = TENSORALLOC_CALLOC;
            // if we've found the dataformat there are no more dimensions
            // check right away if the arity is correct
            if (remaining_args != len && enforceArity == 1) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR wrong number of arguments for 'AI.TENSORSET' command");
                return -1;
            }
            argpos++;
            break;
        } else {
            long long dimension;
            const int retval = RedisModule_StringToLongLong(argv[argpos], &dimension);
            if (retval != REDISMODULE_OK || dimension <= 0) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR invalid or negative value found in tensor shape");
                return -1;
            }
            ndims++;
            dims = array_append(dims, dimension);
            len *= dimension;
        }
    }

    if (datafmt == TENSOR_BLOB) {
        RedisModuleString *rstr = argv[argpos];
        *t = _TensorCreateWithDLDataTypeAndRString(datatype, datasize, dims, ndims, rstr, error);
    } else {
        *t = RAI_TensorCreateWithDLDataType(datatype, dims, ndims, tensorAllocMode);
    }
    if (!(*t)) {
        array_free(dims);
        return -1;
    }

    long i = 0;
    if (datafmt == TENSOR_VALUES) {
        for (; (argpos <= argc - 1) && (i < len); argpos++) {
            if (datatype.code == kDLFloat) {
                double val;
                const int retval = RedisModule_StringToDouble(argv[argpos], &val);
                if (retval != REDISMODULE_OK) {
                    RAI_TensorFree(*t);
                    array_free(dims);
                    RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                    return -1;
                }
                const int retset = RAI_TensorSetValueFromDouble(*t, i, val);
                if (retset == -1) {
                    RAI_TensorFree(*t);
                    array_free(dims);
                    RAI_SetError(error, RAI_ETENSORSET,
                                 "ERR cannot specify values for this datatype");
                    return -1;
                }
            } else {
                long long val;
                const int retval = RedisModule_StringToLongLong(argv[argpos], &val);
                if (retval != REDISMODULE_OK) {
                    RAI_TensorFree(*t);
                    array_free(dims);
                    RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                    return -1;
                }
                const int retset = RAI_TensorSetValueFromLongLong(*t, i, val);
                if (retset == -1) {
                    RAI_TensorFree(*t);
                    array_free(dims);
                    RAI_SetError(error, RAI_ETENSORSET,
                                 "ERR cannot specify values for this datatype");
                    return -1;
                }
            }
            i++;
        }
    }
    array_free(dims);
    return argpos;
}

int RAI_TensorReplyWithValues(RedisModuleCtx *ctx, RAI_Tensor *t) {
    long long ndims = RAI_TensorNumDims(t);
    long long len = 1;
    long long i;
    for (i = 0; i < ndims; i++) {
        len *= RAI_TensorDim(t, i);
    }

    DLDataType dtype = RAI_TensorDataType(t);

    RedisModule_ReplyWithArray(ctx, len);

    if (dtype.code == kDLFloat) {
        double val;
        for (i = 0; i < len; i++) {
            int ret = RAI_TensorGetValueAsDouble(t, i, &val);
            if (ret == REDISMODULE_ERR) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
                return -1;
            }
            RedisModule_ReplyWithDouble(ctx, val);
        }
    } else {
        long long val;
        for (i = 0; i < len; i++) {
            int ret = RAI_TensorGetValueAsLongLong(t, i, &val);
            if (ret == REDISMODULE_ERR) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
                return -1;
            }
            RedisModule_ReplyWithLongLong(ctx, val);
        }
    }

    return 0;
}

uint ParseTensorGetArgs(RAI_Error *err, RedisModuleString **argv, int argc) {
    uint fmt = TENSOR_NONE;
    if (argc < 2 || argc > 4) {
        RAI_SetError(err, RAI_EDAGBUILDER, "wrong number of arguments for 'AI.TENSORGET' command");
        return fmt;
    }
    if (argc == 2) {
        return TENSOR_BLOB | TENSOR_META;
    }
    for (int i = 2; i < argc; i++) {
        const char *fmtstr = RedisModule_StringPtrLen(argv[i], NULL);
        if (!strcasecmp(fmtstr, "BLOB")) {
            fmt |= TENSOR_BLOB;
        } else if (!strcasecmp(fmtstr, "VALUES")) {
            fmt |= TENSOR_VALUES;
        } else if (!strcasecmp(fmtstr, "META")) {
            fmt |= TENSOR_META;
        } else {
            RAI_SetError(err, RAI_EDAGBUILDER, "ERR unsupported data format");
            return TENSOR_NONE;
        }
    }

    if (fmt == TENSOR_ILLEGAL_VALUES_BLOB) {
        RAI_SetError(err, RAI_EDAGBUILDER, "ERR both BLOB and VALUES specified");
        return TENSOR_NONE;
    }
    return fmt;
}

int ReplyWithTensor(RedisModuleCtx *ctx, uint fmt, RAI_Tensor *t) {

    if (!(fmt & TENSOR_META)) {
        if (fmt & TENSOR_BLOB) {
            long long size = RAI_TensorByteSize(t);
            char *data = RAI_TensorData(t);
            RedisModule_ReplyWithStringBuffer(ctx, data, size);
            return REDISMODULE_OK;
        }
        if (fmt & TENSOR_VALUES) {
            int ret = RAI_TensorReplyWithValues(ctx, t);
            if (ret == -1) {
                return REDISMODULE_ERR;
            }
            return REDISMODULE_OK;
        }
    }

    long long resplen = 4;
    if (fmt & (TENSOR_BLOB | TENSOR_VALUES))
        resplen += 2;

    const long long ndims = RAI_TensorNumDims(t);

    char dtypestr[8];
    const int dtypestr_result = Tensor_DataTypeStr(RAI_TensorDataType(t), dtypestr);
    if (dtypestr_result == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
        return REDISMODULE_ERR;
    }

    RedisModule_ReplyWithArray(ctx, resplen);
    RedisModule_ReplyWithCString(ctx, "dtype");
    RedisModule_ReplyWithCString(ctx, dtypestr);
    RedisModule_ReplyWithCString(ctx, "shape");

    RedisModule_ReplyWithArray(ctx, ndims);
    for (int i = 0; i < ndims; i++) {
        const long long dim = RAI_TensorDim(t, i);
        RedisModule_ReplyWithLongLong(ctx, dim);
    }

    if (fmt & TENSOR_BLOB) {
        long long size = RAI_TensorByteSize(t);
        char *data = RAI_TensorData(t);
        RedisModule_ReplyWithCString(ctx, "blob");
        RedisModule_ReplyWithStringBuffer(ctx, data, size);

    } else if (fmt & TENSOR_VALUES) {
        RedisModule_ReplyWithCString(ctx, "values");
        int ret = RAI_TensorReplyWithValues(ctx, t);
        if (ret != REDISMODULE_OK) {
            return REDISMODULE_ERR;
        }
    }
    return REDISMODULE_OK;
}

RedisModuleType *RAI_TensorRedisType(void) { return RedisAI_TensorType; }
