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
    int64_t shape[ndims];
    for (size_t i = 0; i < ndims; ++i) {
        shape[i] = RedisModule_LoadSigned(io);
    }

    RAI_Error err = {0};
    size_t blob_len;
    char *data = RedisModule_LoadStringBuffer(io, &blob_len);
    if (RedisModule_IsIOError(io))
        goto error;
    RAI_Tensor *t =
        RAI_TensorCreateFromBlob(data_type, (const size_t *)shape, ndims, data, blob_len, &err);
    if (t == NULL)
        goto error;

    return t;

error:
    RAI_ClearError(&err);
    RedisModule_LogIOError(io, "error", "Experienced a short read while reading a tensor from RDB");
    return NULL;
}

void *RAI_RDBLoadModel_v4(RedisModuleIO *io) { return RAI_RDBLoadModel_v3(io); }

void *RAI_RDBLoadScript_v4(RedisModuleIO *io) { return RAI_RDBLoadScript_v3(io); }
