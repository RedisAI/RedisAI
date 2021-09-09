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

RedisModuleType *RAI_TensorRedisType(void) { return RedisAI_TensorType; }

//************************************** static helper functions *******************

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

static int _RAI_TensorParseStringValues(int argc, RedisModuleString **argv, RAI_Tensor *tensor,
                                        RAI_Error *err) {
    size_t total_len = 0;
    const char *strings_data[argc];
    size_t strings_lengths[argc];
    uint64_t *strings_offsets = RAI_TensorStringElementsOffsets(tensor);

    // go over the strings and save the offset every string element in the blob.
    for (int i = 0; i < argc; i++) {
        strings_offsets[i] = total_len;
        size_t str_len;
        const char *str_data = RedisModule_StringPtrLen(argv[i], &str_len);
        // Validate that the given string is a C-string (doesn't contain \0 except for the end)
        for (size_t j = 0; j < str_len - 1; j++) {
            if (str_data[j] == '\0') {
                RAI_SetError(err, RAI_ETENSORSET, "ERR C-string value contains null character");
                return REDISMODULE_ERR;
            }
        }
        strings_lengths[i] = str_len;
        total_len += str_len;
        if (str_data[str_len - 1] != '\0') {
            total_len++;
        }
        strings_data[i] = str_data;
    }
    tensor->tensor.dl_tensor.data = RedisModule_Calloc(1, total_len);
    tensor->tensor.dl_tensor.elements_length = strings_offsets;
    tensor->blobSize = total_len;

    // Copy the strings one by one to the data ptr.
    for (int i = 0; i < argc; i++) {
        memcpy(RAI_TensorData(tensor) + strings_offsets[i], strings_data[i], strings_lengths[i]);
    }

    return REDISMODULE_OK;
}

static int _RAI_TensorFillWithValues(int argc, RedisModuleString **argv, RAI_Tensor *t,
                                     DLDataType data_type, RAI_Error *error) {
    t->blobSize = RAI_TensorLength(t) * RAI_TensorDataSize(t);
    t->tensor.dl_tensor.data = RedisModule_Alloc(t->blobSize);

    for (long long i = 0; i < argc; i++) {
        if (data_type.code == kDLFloat) {
            double val;
            int ret_val = RedisModule_StringToDouble(argv[i], &val);
            if (ret_val != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                return REDISMODULE_ERR;
            }
            int ret_set = RAI_TensorSetValueFromDouble(t, i, val);
            if (ret_set != 1) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR cannot specify values for this data type");
                return REDISMODULE_ERR;
            }
        } else {
            long long val;
            const int ret_val = RedisModule_StringToLongLong(argv[i], &val);
            if (ret_val != REDISMODULE_OK || _ValOverflow(val, t)) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR invalid value");
                return REDISMODULE_ERR;
            }
            const int ret_set = RAI_TensorSetValueFromLongLong(t, i, val);
            if (ret_set != 1) {
                RAI_SetError(error, RAI_ETENSORSET, "ERR cannot specify values for this data type");
                return REDISMODULE_ERR;
            }
        }
    }
    return REDISMODULE_OK;
}

static int _RAI_TensorParseBooleansBlob(const char *tensor_blob, size_t blob_len, size_t tensor_len,
                                       RAI_Error *err) {

    // if we encounter a non-boolean value - return an error
    for (size_t i = 0; i < blob_len - 1; i++) {
        if (tensor_blob[i] != 0 && tensor_blob[i] != 1) {
            if (err) {
                RAI_SetError(err, RAI_ETENSORSET,
                             "ERR BOOL tensor elements must be 0 or 1");
            }
            return REDISMODULE_ERR;
        }
    }
    return REDISMODULE_OK;
}

// This will populate the offsets array with the start position of every string element in the blob
static int _RAI_TensorParseStringsBlob(const char *tensor_blob, size_t blob_len, size_t tensor_len,
                                       uint64_t *offsets, RAI_Error *err) {

    size_t elements_counter = 0;
    offsets[elements_counter++] = 0;

    // if we encounter null-character, we set the next element offset to the next position
    for (size_t i = 0; i < blob_len - 1; i++) {
        if (tensor_blob[i] == '\0') {
            offsets[elements_counter++] = i + 1;
        }
    }
    if (tensor_blob[blob_len - 1] != '\0' || elements_counter != tensor_len) {
        if (err) {
            RAI_SetError(err, RAI_ETENSORSET,
                         "ERR Number of string elements in data blob does not match tensor length");
        }
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static int _RAI_TensorReplyWithValues(RedisModuleCtx *ctx, RAI_Tensor *t) {
    int ndims = RAI_TensorNumDims(t);
    long long len = 1;
    for (int i = 0; i < ndims; i++) {
        len *= RAI_TensorDim(t, i);
    }

    DLDataType dtype = RAI_TensorDataType(t);
    RedisModule_ReplyWithArray(ctx, len);

    if (dtype.code == kDLString) {
        for (long long i = 0; i < len; i++) {
            const char *val;
            int ret = RAI_TensorGetValueAsCString(t, i, &val);
            if (!ret) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this data type");
                return REDISMODULE_ERR;
            }
            RedisModule_ReplyWithCString(ctx, val);
        }
    } else if (dtype.code == kDLFloat) {
        double val;
        for (long long i = 0; i < len; i++) {
            int ret = RAI_TensorGetValueAsDouble(t, i, &val);
            if (!ret) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this data type");
                return REDISMODULE_ERR;
            }
            RedisModule_ReplyWithDouble(ctx, val);
        }
    } else {
        long long val;
        for (long long i = 0; i < len; i++) {
            int ret = RAI_TensorGetValueAsLongLong(t, i, &val);
            if (!ret) {
                RedisModule_ReplyWithError(ctx, "ERR cannot get values for this data type");
                return REDISMODULE_ERR;
            }
            RedisModule_ReplyWithLongLong(ctx, val);
        }
    }
    return REDISMODULE_OK;
}

//***************** methods for creating a tensor ************************************

RAI_Tensor *RAI_TensorNew(DLDataType data_type, const size_t *dims, int n_dims) {

    RAI_Tensor *new_tensor = RedisModule_Alloc(sizeof(RAI_Tensor));
    new_tensor->refCount = 1;
    new_tensor->blobSize = 0;

    // Note that n_dim can be zero (i.e., tensor is a scalar)
    int64_t *shape = RedisModule_Calloc(n_dims, sizeof(*shape));
    int64_t *strides = NULL;
    uint64_t *offsets = NULL;

    size_t tensor_len = 1;
    for (int64_t i = 0; i < n_dims; ++i) {
        shape[i] = (int64_t)dims[i];
        tensor_len *= dims[i];
    }
    new_tensor->len = tensor_len;

    if (data_type.code != kDLString) {
        strides = RedisModule_Calloc(n_dims, sizeof(*strides));
        if (n_dims > 0) {
            strides[n_dims - 1] = 1;
        }
        for (int i = n_dims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    } else {
        offsets = RedisModule_Alloc(tensor_len * sizeof(*offsets));
    }

    // Default device is CPU (id -1 is default, means 'no index')
    DLDevice device = (DLDevice){.device_type = kDLCPU, .device_id = -1};

    new_tensor->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.device = device,
                                                                   .data = NULL,
                                                                   .ndim = n_dims,
                                                                   .dtype = data_type,
                                                                   .shape = shape,
                                                                   .strides = strides,
                                                                   .byte_offset = 0,
                                                                   .elements_length = offsets},
                                           .manager_ctx = NULL,
                                           .deleter = NULL};

    return new_tensor;
}

RAI_Tensor *RAI_TensorCreateFromValues(DLDataType data_type, const size_t *dims, int n_dims,
                                       int argc, RedisModuleString **argv, RAI_Error *err) {

    RAI_Tensor *new_tensor = RAI_TensorNew(data_type, dims, n_dims);
    size_t tensor_len = RAI_TensorLength(new_tensor);

    // If no values were given, we consider the empty tensor as a sequence of zeros
    // (null-character in case of string tensor)
    if (argc == 0) {
        new_tensor->blobSize = RAI_TensorDataSize(new_tensor) * tensor_len;
        new_tensor->tensor.dl_tensor.data = RedisModule_Calloc(1, new_tensor->blobSize);
        if (data_type.code == kDLString) {
            uint64_t *strings_offsets = RAI_TensorStringElementsOffsets(new_tensor);
            for (size_t i = 0; i < tensor_len; i++) {
                strings_offsets[i] = i;
            }
        }
        return new_tensor;
    }

    if (data_type.code == kDLString) {
        // Allocate and populate the data blob and save every string element's offset within it.
        if (_RAI_TensorParseStringValues(argc, argv, new_tensor, err) != REDISMODULE_OK) {
            RAI_TensorFree(new_tensor);
            return NULL;
        }
    } else {
        // Parse and fill tensor data with numeric values
        if (_RAI_TensorFillWithValues(argc, argv, new_tensor, data_type, err) != REDISMODULE_OK) {
            RAI_TensorFree(new_tensor);
            return NULL;
        }
    }
    return new_tensor;
}

RAI_Tensor *RAI_TensorCreateFromBlob(DLDataType data_type, const size_t *dims, int n_dims,
                                     const char *tensor_blob, size_t blob_len, RAI_Error *err) {

    RAI_Tensor *new_tensor = RAI_TensorNew(data_type, dims, n_dims);
    size_t tensor_len = RAI_TensorLength(new_tensor);
    size_t data_type_size = RAI_TensorDataSize(new_tensor);

    new_tensor->blobSize = blob_len;
    new_tensor->tensor.dl_tensor.data = RedisModule_Alloc(blob_len);
    if (data_type.code == kDLString) {
        // find and save the offset of every individual string in the blob. Set an error if
        // the number of strings in the given blob doesn't match the tensor length.
        uint64_t *offsets = RAI_TensorStringElementsOffsets(new_tensor);
        if (_RAI_TensorParseStringsBlob(tensor_blob, blob_len, tensor_len, offsets, err) !=
            REDISMODULE_OK) {
            RAI_TensorFree(new_tensor);
            return NULL;
        }
    } else {
        size_t expected_n_bytes = tensor_len * data_type_size;
        if (blob_len != expected_n_bytes) {
            RAI_TensorFree(new_tensor);
            RAI_SetError(err, RAI_ETENSORSET,
                         "ERR data length does not match tensor shape and type");
            return NULL;
        }
        if (data_type.code == kDLBool) {
            if (_RAI_TensorParseBooleansBlob(tensor_blob, blob_len, tensor_len, err) !=
                REDISMODULE_OK) {
                RAI_TensorFree(new_tensor);
                return NULL;
            }
        }
    }

    // Copy the blob. We must copy instead of increasing the ref count since we don't have
    // a way of taking ownership on the underline data pointer (this will require introducing
    // a designated RedisModule API, might be optional in the future)
    memcpy(RAI_TensorData(new_tensor), tensor_blob, blob_len);
    return new_tensor;
}

RAI_Tensor *RAI_TensorCreate(const char *data_type_str, const long long *dims, int n_dims) {
    DLDataType data_type = RAI_TensorDataTypeFromString(data_type_str);
    if (data_type.bits == 0) {
        return NULL;
    }

    // return an empty tensor (with empty list of values).
    return RAI_TensorCreateFromValues(data_type, (const size_t *)dims, n_dims, 0, NULL, NULL);
}

RAI_Tensor *RAI_TensorCreateByConcatenatingTensors(RAI_Tensor **tensors, long long n) {
    RedisModule_Assert(n > 0);

    long long total_batch_size = 0;
    long long batch_sizes[n];
    size_t batch_offsets[n];

    int n_dims = RAI_TensorNumDims(tensors[0]);
    size_t dims[n_dims];

    // TODO: check that all tensors have compatible dims
    // Calculate the total batch size after concatenation, this will be the first dim.
    for (long long i = 0; i < n; i++) {
        batch_sizes[i] = RAI_TensorDim(tensors[i], 0);
        total_batch_size += batch_sizes[i];
    }
    dims[0] = total_batch_size;

    // Get the rest of the tensor's dimensions.
    for (int i = 1; i < n_dims; i++) {
        dims[i] = RAI_TensorDim(tensors[0], i);
    }

    // Create a new tensor to store the concatenated tensor in it.
    DLDataType data_type = RAI_TensorDataType(tensors[0]);
    RAI_Tensor *ret = RAI_TensorNew(data_type, dims, n_dims);

    // get the offsets for every tensor's blob in the batched tensor blob
    batch_offsets[0] = 0;
    for (long long i = 1; i < n; i++) {
        batch_offsets[i] = batch_offsets[i - 1] + RAI_TensorByteSize(tensors[i - 1]);
    }
    ret->blobSize = batch_offsets[n - 1] + RAI_TensorByteSize(tensors[n - 1]);
    ret->tensor.dl_tensor.data = RedisModule_Alloc(ret->blobSize);

    // Copy the input tensors data to the new tensor.
    size_t element_ind = 0;
    uint64_t *ret_offsets = RAI_TensorStringElementsOffsets(ret);
    for (size_t i = 0; i < n; i++) {
        memcpy(RAI_TensorData(ret) + batch_offsets[i], RAI_TensorData(tensors[i]),
               RAI_TensorByteSize(tensors[i]));
        if (data_type.code == kDLString) {
            uint64_t *curr_tensor_offsets = RAI_TensorStringElementsOffsets(tensors[i]);
            for (size_t j = 0; j < RAI_TensorLength(tensors[i]); j++) {
                ret_offsets[element_ind++] = curr_tensor_offsets[j] + batch_offsets[i];
            }
        }
    }
    return ret;
}

RAI_Tensor *RAI_TensorCreateBySlicingTensor(RAI_Tensor *t, long long offset, long long len) {

    RedisModule_Assert(offset >= 0 || len >= 0 || offset + len <= RAI_TensorLength(t));
    int n_dims = RAI_TensorNumDims(t);
    size_t dims[n_dims];

    size_t basic_tensor_len = 1; // the len of every "non-batched" tensor
    for (int i = 1; i < n_dims; i++) {
        dims[i] = RAI_TensorDim(t, i);
        basic_tensor_len *= dims[i];
    }
    dims[0] = len;

    // Create a new tensor to store the sliced tensor in it.
    DLDataType data_type = RAI_TensorDataType(t);
    size_t data_type_size = RAI_TensorDataSize(t);
    RAI_Tensor *ret = RAI_TensorNew(data_type, dims, n_dims);

    // Copy the input tensor sliced data to the new tensor.
    if (data_type.code == kDLString) {
        size_t first_element_pos_in_slice = offset * basic_tensor_len;
        size_t first_element_pos_after_slice = (offset + len) * basic_tensor_len;
        uint64_t *strings_offsets = RAI_TensorStringElementsOffsets(t);
        size_t blob_size =
            first_element_pos_after_slice < RAI_TensorLength(t)
                ? strings_offsets[first_element_pos_after_slice] -
                      strings_offsets[first_element_pos_in_slice]
                : RAI_TensorByteSize(t) - strings_offsets[first_element_pos_in_slice];
        ret->blobSize = blob_size;
        ret->tensor.dl_tensor.data = RedisModule_Alloc(blob_size);
        memcpy(RAI_TensorData(ret), RAI_TensorData(t) + strings_offsets[offset * basic_tensor_len],
               blob_size);
        memcpy(RAI_TensorStringElementsOffsets(ret), strings_offsets + offset,
               len * sizeof(uint64_t));
    } else {
        size_t blob_size = len * basic_tensor_len * data_type_size;
        ret->tensor.dl_tensor.data = RedisModule_Alloc(blob_size);
        memcpy(RAI_TensorData(ret), RAI_TensorData(t) + offset * basic_tensor_len * data_type_size,
               blob_size);
        ret->blobSize = blob_size;
    }
    return ret;
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

RAI_Tensor *RAI_TensorCreateFromDLTensor(DLManagedTensor *dl_tensor) {

    RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(RAI_Tensor));
    ret->refCount = 1;
    ret->tensor = *dl_tensor;         // shallow copy, takes ownership on the dl_tensor memory.
    ret->len = RAI_TensorLength(ret); // compute and set the length
    ret->blobSize = RAI_TensorByteSize(ret);
    return ret;
}

//******************** getters and setters ****************************

int RAI_TensorGetDataTypeStr(DLDataType data_type, char *data_type_str) {
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
    } else if (data_type.code == kDLString) {
        strcpy(data_type_str, RAI_DATATYPE_STR_STRING);
        result = REDISMODULE_OK;
    }
    return result;
}

DLTensor *RAI_TensorGetDLTensor(RAI_Tensor *tensor) { return &tensor->tensor.dl_tensor; }

DLDataType RAI_TensorDataType(RAI_Tensor *t) { return t->tensor.dl_tensor.dtype; }

size_t RAI_TensorLength(RAI_Tensor *t) {
    if (t->len == 0) {
        int64_t *shape = t->tensor.dl_tensor.shape;
        size_t len = 1;
        for (size_t i = 0; i < t->tensor.dl_tensor.ndim; ++i) {
            len *= shape[i];
        }
        t->len = len;
    }
    return t->len;
}

size_t RAI_TensorDataSize(RAI_Tensor *t) { return RAI_TensorDataType(t).bits / 8; }

int RAI_TensorNumDims(RAI_Tensor *t) { return t->tensor.dl_tensor.ndim; }

long long RAI_TensorDim(RAI_Tensor *t, int i) { return t->tensor.dl_tensor.shape[i]; }

size_t RAI_TensorByteSize(RAI_Tensor *t) {

    if (t->tensor.dl_tensor.dtype.code == kDLString) {
        return t->blobSize;
    }
    if (t->blobSize == 0) {
        t->blobSize = RAI_TensorLength(t) * RAI_TensorDataSize(t);
    }
    return t->blobSize;
}

char *RAI_TensorData(RAI_Tensor *t) { return t->tensor.dl_tensor.data; }

uint64_t *RAI_TensorStringElementsOffsets(RAI_Tensor *tensor) {
    return tensor->tensor.dl_tensor.elements_length;
}

int64_t *RAI_TensorShape(RAI_Tensor *tensor) { return tensor->tensor.dl_tensor.shape; }

int RAI_TensorGetValueAsDouble(RAI_Tensor *t, long long i, double *val) {
    // Validate that i is in bound
    if (i < 0 || i > RAI_TensorLength(t)) {
        return 0;
    }
    DLDataType dtype = RAI_TensorDataType(t);
    void *data = RAI_TensorData(t);

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

    // Validate that i is in bound
    if (i < 0 || i > RAI_TensorLength(t)) {
        return 0;
    }
    DLDataType dtype = RAI_TensorDataType(t);
    void *data = RAI_TensorData(t);

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

int RAI_TensorGetValueAsCString(RAI_Tensor *t, long long i, const char **val) {
    // Validate that i is in bound
    if (i < 0 || i > RAI_TensorLength(t)) {
        return 0;
    }
    DLDataType dtype = RAI_TensorDataType(t);
    char *data = RAI_TensorData(t);

    if (dtype.code != kDLString) {
        return 0;
    }
    uint64_t *offsets = RAI_TensorStringElementsOffsets(t);
    char *str_val = data + offsets[i];
    *val = str_val;
    return 1;
}

int RAI_TensorSetData(RAI_Tensor *t, const char *data, size_t len) {
    DLDataType data_type = RAI_TensorDataType(t);
    if (data_type.code == kDLString) {
        if (_RAI_TensorParseStringsBlob(data, len, RAI_TensorLength(t),
                                        RAI_TensorStringElementsOffsets(t),
                                        NULL) != REDISMODULE_OK) {
            return 0;
        }
        RedisModule_Free(RAI_TensorData(t));
        t->tensor.dl_tensor.data = RedisModule_Alloc(len);
    } else if (data_type.code == kDLBool) {
        if (_RAI_TensorParseBooleansBlob(data, len, RAI_TensorLength(t),
                                        NULL) != REDISMODULE_OK) {
            return 0;
        }
    }
    memcpy(RAI_TensorData(t), data, len);
    t->blobSize = len;
    return 1;
}

int RAI_TensorSetValueFromLongLong(RAI_Tensor *t, long long i, long long val) {
    DLDataType dtype = RAI_TensorDataType(t);
    void *data = RAI_TensorData(t);

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
    DLDataType dtype = RAI_TensorDataType(t);
    void *data = RAI_TensorData(t);

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

//******************** tensor memory management *********************************

RAI_Tensor *RAI_TensorGetShallowCopy(RAI_Tensor *t) {
    __atomic_fetch_add(&t->refCount, 1, __ATOMIC_RELAXED);
    return t;
}

void RAI_TensorFree(RAI_Tensor *t) {
    if (t == NULL) {
        return;
    }
    long long ref_count = __atomic_sub_fetch(&t->refCount, 1, __ATOMIC_RELAXED);
    RedisModule_Assert(ref_count >= 0);
    if (ref_count > 0) {
        return;
    }
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
        if (t->tensor.dl_tensor.elements_length) {
            RedisModule_Free(t->tensor.dl_tensor.elements_length);
        }
        RedisModule_Free(t);
    }
}

//***************************** retrieve tensor from keyspace *********************

int RAI_TensorOpenKey(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
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

int RAI_TensorGetFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
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

void RAI_TensorReplicate(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t) {
    long long n_dims = RAI_TensorNumDims(t);

    char data_type_str[8];
    int status = RAI_TensorGetDataTypeStr(RAI_TensorDataType(t), data_type_str);
    RedisModule_Assert(status == REDISMODULE_OK);

    char *data = RAI_TensorData(t);
    long long size = (long long)RAI_TensorByteSize(t);

    RedisModuleString *dims[n_dims];
    for (int i = 0; i < n_dims; i++) {
        dims[i] = RedisModule_CreateStringFromLongLong(NULL, RAI_TensorDim(t, i));
    }

    RedisModule_Replicate(ctx, "AI.TENSORSET", "scvcb", key, data_type_str, dims, n_dims, "BLOB",
                          data, size);

    for (long long i = 0; i < n_dims; i++) {
        RedisModule_FreeString(NULL, dims[i]);
    }
}

int RAI_TensorReply(RedisModuleCtx *ctx, uint fmt, RAI_Tensor *t) {

    if (!(fmt & TENSOR_META)) {
        if (fmt & TENSOR_BLOB) {
            size_t size = RAI_TensorByteSize(t);
            char *data = RAI_TensorData(t);
            RedisModule_ReplyWithStringBuffer(ctx, data, size);
            return REDISMODULE_OK;
        }
        if (fmt & TENSOR_VALUES) {
            return _RAI_TensorReplyWithValues(ctx, t);
        }
    }

    long long resp_len = 4;
    if (fmt & (TENSOR_BLOB | TENSOR_VALUES))
        resp_len += 2;

    const long long n_dims = RAI_TensorNumDims(t);

    char data_type_str[8];
    const int data_type_str_result = RAI_TensorGetDataTypeStr(RAI_TensorDataType(t), data_type_str);
    if (data_type_str_result == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
        return REDISMODULE_ERR;
    }

    RedisModule_ReplyWithArray(ctx, resp_len);
    RedisModule_ReplyWithCString(ctx, "dtype");
    RedisModule_ReplyWithCString(ctx, data_type_str);
    RedisModule_ReplyWithCString(ctx, "shape");

    RedisModule_ReplyWithArray(ctx, n_dims);
    for (int i = 0; i < n_dims; i++) {
        const long long dim = RAI_TensorDim(t, i);
        RedisModule_ReplyWithLongLong(ctx, dim);
    }

    if (fmt & TENSOR_BLOB) {
        size_t size = RAI_TensorByteSize(t);
        char *data = RAI_TensorData(t);
        RedisModule_ReplyWithCString(ctx, "blob");
        RedisModule_ReplyWithStringBuffer(ctx, data, size);

    } else if (fmt & TENSOR_VALUES) {
        RedisModule_ReplyWithCString(ctx, "values");
        return _RAI_TensorReplyWithValues(ctx, t);
    }
    return REDISMODULE_OK;
}
