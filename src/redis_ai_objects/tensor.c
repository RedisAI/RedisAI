/**
 * tensor.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisAI_TensorType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <strings.h>
#include "tensor.h"
#include "err.h"
#include "arr.h"
#include "redisai.h"
#include "version.h"
#include "tensor_struct.h"
#include "rmutil/alloc.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include "execution/utils.h"

extern RedisModuleType *RedisAI_TensorType;

// Check if the given value is in the range of the tensor type.
static bool _ValOverflow(long long val, RAI_Tensor *t) {
    DLDataType dtype = t->tensor.dl_tensor.dtype;
    if (dtype.code == kDLInt) {
        unsigned long long max_abs_val = ((unsigned long long)1 << (uint)(dtype.bits - 1));
        if ((unsigned long long)val >= max_abs_val || val < -1 * (long long)max_abs_val) {
            return true;
        }
    } else if (dtype.code == kDLUInt) {
        uint max_val = (uint)1 << dtype.bits;
        if (val >= max_val || val < 0) {
            return true;
        }
    } else if (dtype.code == kDLBool) {
        if (val < 0 || val > 1) {
            return true;
        }
    }
    return false;
}

static size_t _RAI_StringTensorGetDataLen(size_t arg_pos, int argc, RedisModuleString **argv) {
    size_t total_len = 0;
    for (; arg_pos < argc; arg_pos++) {
        size_t str_len;
        RedisModule_StringPtrLen(argv[arg_pos], &str_len);
        total_len +=
    }
}

static int _RAI_TensorFillWithValues(int arg_pos, int argc, RedisModuleString **argv, RAI_Tensor *t, long long len,
                                 DLDataType data_type, RAI_Error *error) {
    for (long long i = 0; (arg_pos < argc) && (i < len); arg_pos++, i++) {
        if (data_type.code == kDLFloat) {
            double val;
            const int ret_val = RedisModule_StringToDouble(argv[arg_pos], &val);
            if (ret_val != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                return REDISMODULE_ERR;
            }
            const int ret_set = RAI_TensorSetValueFromDouble(t, i, val);
            if (ret_set == -1) {
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR cannot specify values for this datatype");
                return REDISMODULE_ERR;
            }
        } else {
            long long val;
            const int ret_val = RedisModule_StringToLongLong(argv[arg_pos], &val);
            if (ret_val != REDISMODULE_OK || _ValOverflow(val, t)) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                return REDISMODULE_ERR;
            }
            const int ret_set = RAI_TensorSetValueFromLongLong(t, i, val);
            if (ret_set == -1) {
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR cannot specify values for this data type");
                return REDISMODULE_ERR;
            }
        }
    }
    return REDISMODULE_OK;
}

static uint64_t *_RAI_TensorGetStringsOffsets(const char *tensor_blob, size_t blob_len,
                                              size_t tensor_len, RAI_Error *err) {
    uint64_t *strings_offsets = RedisModule_Alloc(tensor_len * sizeof(*strings_offsets));
    size_t elements_counter = 0;
    strings_offsets[0] = 0;
    for (size_t i = 0; i < blob_len-1; i++) {
        if (tensor_blob[i] == '\0') {
            strings_offsets[elements_counter++] = i+1;
        }
    }
    if (tensor_blob[blob_len-1] != '\0' || elements_counter != tensor_len) {
        RAI_SetError(err, RAI_ETENSORSET, "ERR String tensor blob must contain a sequence of null-terminated"
                                          " strings whose size is the number of elements in the tensor");
        RedisModule_Free(strings_offsets);
        return NULL;
    }
    return strings_offsets;
}

DLDataType RAI_TensorDataTypeFromString(const char *type_str) {
    if (strcasecmp(type_str, RAI_DATATYPE_STR_FLOAT) == 0) {
        return (DLDataType){.code = kDLFloat, .bits = 32, .lanes = 1};
    }
    if (strcasecmp(type_str, RAI_DATATYPE_STR_DOUBLE) == 0) {
        return (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1};
    }
    if (strncasecmp(type_str, "INT", 3) == 0) {
        const char *bit_str = type_str + 3;
        if (strcmp(bit_str, "8") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 8, .lanes = 1};
        }
        if (strcmp(bit_str, "16") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 16, .lanes = 1};
        }
        if (strcmp(bit_str, "32") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
        }
        if (strcmp(bit_str, "64") == 0) {
            return (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
        }
    }
    if (strncasecmp(type_str, "UINT", 4) == 0) {
        const char *bit_str = type_str + 4;
        if (strcmp(bit_str, "8") == 0) {
            return (DLDataType){.code = kDLUInt, .bits = 8, .lanes = 1};
        }
        if (strcmp(bit_str, "16") == 0) {
            return (DLDataType){.code = kDLUInt, .bits = 16, .lanes = 1};
        }
    }
    if (strcasecmp(type_str, "BOOL") == 0) {
        return (DLDataType){.code = kDLBool, .bits = 8, .lanes = 1};
    }
    if (strcasecmp(type_str, "STRING") == 0) {
        return (DLDataType){.code = kDLString, .bits = 8, .lanes = 1};
    }
    // Invalid data type
    return (DLDataType){.bits = 0};
}

static size_t _RAI_TensorDataTypeSize(DLDataType data_type) { return data_type.bits / 8; }

int Tensor_DataTypeStr(DLDataType data_type, char *data_type_str) {
    int result = REDISMODULE_ERR;

    if (data_type.code == kDLFloat) {
        if (data_type.bits == 32) {
            strcpy(data_type_str, RAI_DATATYPE_STR_FLOAT);
            result = REDISMODULE_OK;
        } else if (data_type.bits == 64) {
            strcpy(data_type_str, RAI_DATATYPE_STR_DOUBLE);
            result = REDISMODULE_OK;
        }
    } else if (data_type.code == kDLInt) {
        if (data_type.bits == 8) {
            strcpy(data_type_str, RAI_DATATYPE_STR_INT8);
            result = REDISMODULE_OK;
        } else if (data_type.bits == 16) {
            strcpy(data_type_str, RAI_DATATYPE_STR_INT16);
            result = REDISMODULE_OK;
        } else if (data_type.bits == 32) {
            strcpy(data_type_str, RAI_DATATYPE_STR_INT32);
            result = REDISMODULE_OK;
        } else if (data_type.bits == 64) {
            strcpy(data_type_str, RAI_DATATYPE_STR_INT64);
            result = REDISMODULE_OK;
        }
    } else if (data_type.code == kDLUInt) {
        if (data_type.bits == 8) {
            strcpy(data_type_str, RAI_DATATYPE_STR_UINT8);
            result = REDISMODULE_OK;
        } else if (data_type.bits == 16) {
            strcpy(data_type_str, RAI_DATATYPE_STR_UINT16);
            result = REDISMODULE_OK;
        }
    } else if (data_type.code == kDLBool && data_type.bits == 8) {
        strcpy(data_type_str, RAI_DATATYPE_STR_BOOL);
        result = REDISMODULE_OK;
    } else if (data_type.code == kDLString && data_type.bits == 8) {
        strcpy(data_type_str, RAI_DATATYPE_STR_STRING);
        result = REDISMODULE_OK;
    }

    return result;
}

RAI_Tensor *RAI_TensorNew(void) {
    RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->refCount = 1;
    ret->len = LEN_UNKNOWN;
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
static RAI_Tensor *_RAI_TensorCreateFromValues(const char *type_str, const long long *dims, size_t n_dims,
                                         bool empty, size_t arg_pos, int argc, RedisModuleString **argv) {

    RAI_Tensor *new_tensor = RAI_TensorCreate(type_str, dims, (int)n_dims);
    RedisModule_Assert(new_tensor);

    // If we return an empty tensor, we initialize the data with zeros to avoid security
    // issues. Otherwise, we only allocate without initializing (for better performance).
    uint64_t *strings_offsets = NULL;
    if (empty) {
        memset(new_tensor->tensor.dl_tensor.data, 0, )


        data = RedisModule_Calloc(tensor_len, data_type_size);
        // We consider the empty string tensor as a tensor of empty strings (a sequence of null characters)
        if (data_type.code == kDLString) {
            strings_offsets = RedisModule_Alloc(tensor_len*sizeof(*strings_offsets));
            for (size_t i = 0; i < tensor_len; i++) {
                strings_offsets[i] = i;
            }
        }
    } else {
        if (data_type.code == kDLString) {
            size_t data_length = 0;
            for (size_t i = 0; i < )
        }
        data = RedisModule_Alloc(tensor_len * data_type_size);
    }

    tensor
}

static RAI_Tensor *_RAI_TensorCreateFromBlob(DLDataType data_type, size_t data_type_size,
                                                  const long long *dims, size_t n_dims,
                                                  RedisModuleString *tensor_blob_rs, RAI_Error *err) {

    int64_t *shape = RedisModule_Alloc(n_dims * sizeof(*shape));

    size_t tensor_len = 1;
    for (int64_t i = 0; i < n_dims; ++i) {
        shape[i] = dims[i];
        tensor_len *= dims[i];
    }

    int64_t *strides = NULL;
    if (data_type.code != kDLString) {
        strides = RedisModule_Alloc(n_dims * sizeof(*strides));
        strides[n_dims-1] = 1;
        for (size_t i = n_dims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // Default device is CPU (id -1 is default, means 'no index')
    DLDevice device = (DLDevice){.device_type = kDLCPU, .device_id = -1};

    size_t blob_len;
    const char *tensor_blob = RedisModule_StringPtrLen(tensor_blob_rs, &blob_len);
    uint64_t *offsets = NULL;
    if (data_type.code == kDLString) {
        // get the length of every individual string in the tensor, set an error and return NULL if the number of
        // strings in the given blob doesn't match the number of expected elements in the tensor.
        offsets = _RAI_TensorGetStringsOffsets(tensor_blob, blob_len, tensor_len, err);
        if (offsets == NULL) {
            RedisModule_Free(shape);
            return NULL;
        }
    } else {
        size_t expected_n_bytes = tensor_len * data_type_size;
        if (blob_len != expected_n_bytes) {
            RedisModule_Free(shape);
            RedisModule_Free(strides);
            RAI_SetError(err, RAI_ETENSORSET, "ERR data length does not match tensor shape and type");
            return NULL;
        }
    }
    char *data = RedisModule_Alloc(blob_len);
    memcpy(data, tensor_blob, blob_len);
    RAI_HoldString(tensor_blob_rs);

    RAI_Tensor *ret = RAI_TensorNew();
    ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.device = device,
                                                            .data = data,
                                                            .ndim = (int)n_dims,
                                                            .dtype = data_type,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = 0,
                                                            .elements_length = offsets},
                                    .manager_ctx = tensor_blob_rs,
                                    .deleter = RAI_RStringDataTensorDeleter};

    return ret;
}

RAI_Tensor *RAI_TensorCreate(const char *dataType, const long long *dims, int n_dims) {
    DLDataType data_type = RAI_TensorDataTypeFromString(dataType);
    size_t data_type_size = _RAI_TensorDataTypeSize(data_type);
    if (data_type_size == 0) {
        return NULL;
    }

    RAI_Tensor *new_tensor = RAI_TensorNew();
    int64_t *shape = RedisModule_Alloc(n_dims * sizeof(*shape));
    int64_t *strides = NULL;

    size_t tensor_len = 1;
    for (int64_t i = 0; i < n_dims; ++i) {
        shape[i] = dims[i];
        tensor_len *= dims[i];
    }

    if (data_type.code != kDLString) {
        strides = RedisModule_Alloc(n_dims * sizeof(*strides));
        strides[n_dims-1] = 1;
        for (int64_t i = n_dims - 2; i >= 0; --i) {
            strides[i] *= strides[i + 1] * shape[i + 1];
        }
    }

    // Default device is CPU (id -1 is default, means 'no index')
    DLDevice device = (DLDevice){.device_type = kDLCPU, .device_id = -1};

    void *data = NULL;
    if (data_type.code != kDLString) {
        data = RedisModule_Alloc(tensor_len * data_type_size);
    }

    new_tensor->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.device = device,
                                                            .data = data,
                                                            .ndim = (int)n_dims,
                                                            .dtype = data_type,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = 0,
                                                            .elements_length = NULL},
                                    .manager_ctx = NULL,
                                    .deleter = NULL};

    return new_tensor;
}

RAI_Tensor *RAI_TensorCreateByConcatenatingTensors(RAI_Tensor **ts, long long n) {

    if (n == 0) {
        return NULL;
    }

    long long total_batch_size = 0;
    long long batch_sizes[n];
    long long batch_offsets[n];

    const int ndims = RAI_TensorNumDims(ts[0]);
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

    RAI_Tensor *ret = _RAI_TensorCreateFromValues(dtype, dims, ndims, false);

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

    RAI_Tensor *ret = _RAI_TensorCreateFromValues(dtype, dims, ndims, false);

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

    RAI_Tensor *ret = _RAI_TensorCreateFromValues(dtype, dims, ndims, false);

    memcpy(RAI_TensorData(ret), RAI_TensorData(t), sample_size * dtype_size);
    *dest = ret;
    return 0;
}

// Beware: this will take ownership of dltensor.
RAI_Tensor *RAI_TensorCreateFromDLTensor(DLManagedTensor *dl_tensor) {

    RAI_Tensor *ret = RAI_TensorNew();
    ret->tensor =
        (DLManagedTensor){.dl_tensor = (DLTensor){.device = dl_tensor->dl_tensor.device,
                                                  .data = dl_tensor->dl_tensor.data,
                                                  .ndim = dl_tensor->dl_tensor.ndim,
                                                  .dtype = dl_tensor->dl_tensor.dtype,
                                                  .shape = dl_tensor->dl_tensor.shape,
                                                  .strides = dl_tensor->dl_tensor.strides,
                                                  .byte_offset = dl_tensor->dl_tensor.byte_offset},
                          .manager_ctx = dl_tensor->manager_ctx,
                          .deleter = dl_tensor->deleter};
    RAI_TensorLength(ret); // This will set ret->len field.
    return ret;
}

DLTensor *RAI_TensorGetDLTensor(RAI_Tensor *tensor) { return &tensor->tensor.dl_tensor; }

DLDataType RAI_TensorDataType(RAI_Tensor *t) { return t->tensor.dl_tensor.dtype; }

int RAI_TensorIsDataTypeEqual(RAI_Tensor *t1, RAI_Tensor *t2) {
    return t1->tensor.dl_tensor.dtype.bits == t2->tensor.dl_tensor.dtype.bits &&
           t1->tensor.dl_tensor.dtype.code == t2->tensor.dl_tensor.dtype.code &&
           t1->tensor.dl_tensor.dtype.lanes == t2->tensor.dl_tensor.dtype.lanes;
}

size_t RAI_TensorLength(RAI_Tensor *t) {
    if (t->len == LEN_UNKNOWN && ()) {
        int64_t *shape = t->tensor.dl_tensor.shape;
        size_t len = 1;
        for (size_t i = 0; i < t->tensor.dl_tensor.ndim; ++i) {
            len *= shape[i];
        }
        t->len = len;
    }
    return t->len;
}

size_t RAI_TensorDataSize(RAI_Tensor *t) { return _RAI_TensorDataTypeSize(RAI_TensorDataType(t)); }

size_t RAI_TensorDataSizeFromString(const char *dataTypeStr) {
    DLDataType dtype = RAI_TensorDataTypeFromString(dataTypeStr);
    return _RAI_TensorDataTypeSize(dtype);
}

size_t RAI_TensorDataSizeFromDLDataType(DLDataType dtype) { return _RAI_TensorDataTypeSize(dtype); }

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
        case 16:
            ((int16_t *)data)[i] = val;
            break;
        case 32:
            ((int32_t *)data)[i] = val;
            break;
        case 64:
            ((int64_t *)data)[i] = val;
            break;
        default:
            return 0;
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            ((uint8_t *)data)[i] = val;
            break;
        case 16:
            ((uint16_t *)data)[i] = val;
            break;
        case 32:
            ((uint32_t *)data)[i] = val;
            break;
        case 64:
            ((uint64_t *)data)[i] = val;
            break;
        default:
            return 0;
        }
    } else if (dtype.code == kDLBool) {
        if (dtype.bits == 8) {
            ((uint8_t *)data)[i] = val;
        } else {
            return 0;
        }
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
            return 0;
        }
    } else {
        return 0;
    }
    return 1;
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
            return 0;
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
            return 0;
        }
    } else if (dtype.code == kDLBool) {
        if (dtype.bits == 8) {
            *val = ((uint8_t *)data)[i];
        } else {
            return 0;
        }
    }
    return 1;
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
    return _RAI_TensorDataTypeSize(RAI_TensorDataType(t)) * RAI_TensorLength(t);
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
        if (VerifyKeyInThisShard(ctx, keyName)) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EKEYEMPTY, "ERR tensor key is empty or in a different shard");
        } else {
            RAI_SetError(err, RAI_EKEYEMPTY,
                         "ERR CROSSSLOT Tensor key in request don't hash to the same slot");
        }
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RedisModule_CloseKey(*key);
        RedisModule_Log(ctx, "warning", "%s is not a tensor",
                        RedisModule_StringPtrLen(keyName, NULL));
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
    long long size = (long long)RAI_TensorByteSize(t);

    RedisModuleString *dims[ndims];

    for (int i = 0; i < ndims; i++) {
        dims[i] = RedisModule_CreateStringFromLongLong(ctx, RAI_TensorDim(t, i));
    }

    RedisModule_Replicate(ctx, "AI.TENSORSET", "scvcb", key, dtypestr, dims, ndims, "BLOB", data,
                          size);

    for (long long i = 0; i < ndims; i++) {
        RedisModule_FreeString(ctx, dims[i]);
    }
}

int RAI_TensorSetParseArgs(RedisModuleString **argv, int argc, RAI_Tensor **t, RAI_Error *error) {
    if (argc < 4) {
        RAI_SetError(error, RAI_ETENSORSET, "wrong number of arguments for 'AI.TENSORSET' command");
        return REDISMODULE_ERR;
    }

    // get the tensor data type
    const char *type_str = RedisModule_StringPtrLen(argv[2], NULL);
    DLDataType data_type = RAI_TensorDataTypeFromString(type_str);
    size_t data_size = _RAI_TensorDataTypeSize(data_type);
    if (data_size == 0) {
        RAI_SetError(error, RAI_ETENSORSET, "ERR invalid data type");
        return REDISMODULE_ERR;
    }

    int data_fmt = TENSOR_NONE;
    size_t n_dims = 0;
    long long len = 1;
    long long *dims = (long long *)array_new(long long, 1);
    size_t arg_pos = 3;

    // go over remaining tensor args (shape and data) after parsing its type.
    for (; arg_pos < argc ; arg_pos++) {
        const char *opt = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        if (!strcasecmp(opt, "BLOB")) {
            data_fmt = TENSOR_BLOB;
            // if we've found the data format, then there are no more dimensions
            // check right away if the arity is correct
            size_t remaining_args = argc - 1 - arg_pos;
            if (remaining_args != 1) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR a single binary string should come after the BLOB argument in 'AI.TENSORSET' command");
                return REDISMODULE_ERR;
            }
            arg_pos++;
            break;
        } else if (!strcasecmp(opt, "VALUES")) {
            data_fmt = TENSOR_VALUES;
            // if we've found the data_format, then there are no more dimensions
            // check right away if the arity is correct
            size_t remaining_args = argc - 1 - arg_pos;
            if (remaining_args != len) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR wrong number of values was given in 'AI.TENSORSET' command");
                return REDISMODULE_ERR;
            }
            arg_pos++;
            break;
        } else {
            long long dimension;
            const int ret_val = RedisModule_StringToLongLong(argv[arg_pos], &dimension);
            if (ret_val != REDISMODULE_OK || dimension <= 0) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR invalid or negative value found in tensor shape");
                return REDISMODULE_ERR;
            }
            n_dims++;
            dims = array_append(dims, dimension);
            len *= dimension;
        }
    }

    if (data_fmt == TENSOR_BLOB) {
        RedisModuleString *tensor_blob = argv[arg_pos];
        *t = _RAI_TensorCreateFromBlob(data_type, dims, n_dims, tensor_blob, error);
    } else {
        // If no data was given, we consider the tensor data as a sequence of zeros.
        bool is_empty = (data_fmt == TENSOR_NONE);
        *t = _RAI_TensorCreateFromValues(data_type, dims, n_dims, arg_pos, argc, argv, is_empty);
    }
    if (!(*t)) {
        array_free(dims);
        return REDISMODULE_ERR;
    }

    if (data_fmt == TENSOR_VALUES) {
        if (_RAI_TensorFillWithValues(arg_pos, argc, argv, *t, len, data_type, error) != REDISMODULE_OK) {
            return REDISMODULE_ERR;
        }
    }

    array_free(dims);
    return REDISMODULE_OK;
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
            if (!ret) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
                return -1;
            }
            RedisModule_ReplyWithDouble(ctx, val);
        }
    } else {
        long long val;
        for (i = 0; i < len; i++) {
            int ret = RAI_TensorGetValueAsLongLong(t, i, &val);
            if (!ret) {
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
