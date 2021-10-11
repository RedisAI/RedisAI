#include <strings.h>
#include "tensor_commands_parsing.h"
#include "redis_ai_objects/tensor.h"
#include "util/arr.h"

int ParseTensorSetArgs(RedisModuleString **argv, int argc, RAI_Tensor **t, RAI_Error *error) {
    if (argc < 4) {
        RAI_SetError(error, RAI_ETENSORSET, "wrong number of arguments for 'AI.TENSORSET' command");
        return REDISMODULE_ERR;
    }

    // get the tensor data type
    const char *type_str = RedisModule_StringPtrLen(argv[2], NULL);
    DLDataType data_type = RAI_TensorDataTypeFromString(type_str);
    if (data_type.bits == 0) {
        RAI_SetError(error, RAI_ETENSORSET, "ERR invalid data type");
        return REDISMODULE_ERR;
    }

    int data_fmt = TENSOR_NONE;
    int n_dims = 0;
    long long tensor_len = 1;
    size_t *dims = array_new(size_t, 1);
    int arg_pos = 3;

    // go over remaining tensor args (shapes and data) after parsing its type.
    for (; arg_pos < argc; arg_pos++) {
        const char *opt = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        if (!strcasecmp(opt, "BLOB")) {
            data_fmt = TENSOR_BLOB;
            // if we've found the data format, then there are no more dimensions
            // check right away if the arity is correct
            size_t remaining_args = argc - 1 - arg_pos;
            if (remaining_args != 1) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR a single binary string should come after the BLOB argument in "
                             "'AI.TENSORSET' command");
                return REDISMODULE_ERR;
            }
            arg_pos++;
            break;
        } else if (!strcasecmp(opt, "VALUES")) {
            data_fmt = TENSOR_VALUES;
            // if we've found the data_format, then there are no more dimensions
            // check right away if the arity is correct
            size_t remaining_args = argc - 1 - arg_pos;
            if (remaining_args != tensor_len) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR wrong number of values was given in 'AI.TENSORSET' command");
                return REDISMODULE_ERR;
            }
            arg_pos++;
            break;
        } else {
            // Otherwise, parse the next tensor shape and append it to its dims.
            long long dimension;
            int ret_val = RedisModule_StringToLongLong(argv[arg_pos], &dimension);
            if (ret_val != REDISMODULE_OK || dimension <= 0) {
                array_free(dims);
                RAI_SetError(error, RAI_ETENSORSET,
                             "ERR invalid or negative value found in tensor shape");
                return REDISMODULE_ERR;
            }
            n_dims++;
            dims = array_append(dims, dimension);
            tensor_len *= dimension;
        }
    }

    if (data_fmt == TENSOR_BLOB) {
        RedisModuleString *tensor_blob_rs = argv[arg_pos];
        size_t blob_len;
        const char *tensor_blob = RedisModule_StringPtrLen(tensor_blob_rs, &blob_len);
        *t = RAI_TensorCreateFromBlob(data_type, dims, n_dims, tensor_blob, blob_len, error);
    } else {
        // Parse the rest of the arguments (tensor values) and set the values in the tensor.
        // Note that it is possible that no values were given - create empty tensor in that case.
        *t = RAI_TensorCreateFromValues(data_type, dims, n_dims, argc - arg_pos, &argv[arg_pos],
                                        error);
    }
    array_free(dims);

    if (*t == NULL) {
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

uint ParseTensorGetFormat(RAI_Error *err, RedisModuleString **argv, int argc) {
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
