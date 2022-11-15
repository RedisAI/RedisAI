/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v4.h"
#include "../../previous/v3/decode_v3.h"
#include "assert.h"

/**
 * In case of IO errors, the default return values are:
 * numbers - 0
 * strings - null
 * So only when it is necessary check for IO errors.
 */

void *RAI_RDBLoadTensor_v4(RedisModuleIO *io) {
    DLDataTypeCode code = RedisModule_LoadUnsigned(io);
    uint8_t bits = RedisModule_LoadUnsigned(io);
    DLDataType data_type = (DLDataType){.code = code, .bits = bits, .lanes = 1};

    int ndims = (int)RedisModule_LoadSigned(io);
    size_t shape[ndims];
    for (size_t i = 0; i < ndims; ++i) {
        shape[i] = RedisModule_LoadSigned(io);
    }

    RAI_Tensor *tensor = RAI_TensorNew(data_type, shape, ndims);

    size_t blob_len;
    char *data = RedisModule_LoadStringBuffer(io, &blob_len);
    if (RedisModule_IsIOError(io))
        goto error;

    tensor->blobSize = blob_len;
    tensor->tensor.dl_tensor.data = data;

    if (data_type.code == kDLString) {
        for (size_t i = 0; i < RAI_TensorLength(tensor); i++) {
            tensor->tensor.dl_tensor.elements_length[i] = RedisModule_LoadUnsigned(io);
        }
    }
    if (RedisModule_IsIOError(io))
        goto error;
    return tensor;

error:
    RedisModule_LogIOError(io, "error", "Experienced a short read while reading a tensor from RDB");
    RAI_TensorFree(tensor);
    if (data) {
        RedisModule_Free(data);
    }
    return NULL;
}

void *RAI_RDBLoadModel_v4(RedisModuleIO *io) { return RAI_RDBLoadModel_v3(io); }

void *RAI_RDBLoadScript_v4(RedisModuleIO *io) { return RAI_RDBLoadScript_v3(io); }
