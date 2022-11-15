/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v4.h"

void RAI_RDBSaveTensor_v4(RedisModuleIO *io, void *value) {
    RAI_Tensor *tensor = (RAI_Tensor *)value;

    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.code);
    RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.dtype.bits);

    size_t ndim = tensor->tensor.dl_tensor.ndim;
    RedisModule_SaveSigned(io, ndim);
    for (size_t i = 0; i < ndim; i++) {
        RedisModule_SaveSigned(io, tensor->tensor.dl_tensor.shape[i]);
    }

    size_t size = RAI_TensorByteSize(tensor);
    RedisModule_SaveStringBuffer(io, tensor->tensor.dl_tensor.data, size);

    if (tensor->tensor.dl_tensor.dtype.code == kDLString) {
        for (size_t i = 0; i < RAI_TensorLength(tensor); i++) {
            RedisModule_SaveUnsigned(io, tensor->tensor.dl_tensor.elements_length[i]);
        }
    }
}

void RAI_RDBSaveModel_v4(RedisModuleIO *io, void *value) {
    RAI_Model *model = (RAI_Model *)value;
    char *buffer = NULL;
    size_t len = 0;
    RAI_Error err = {0};

    int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

    if (err.code != RAI_OK) {
        RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
        if (buffer) {
            RedisModule_Free(buffer);
        }
        return;
    }

    RedisModule_SaveUnsigned(io, model->backend);
    RedisModule_SaveStringBuffer(io, model->devicestr, strlen(model->devicestr) + 1);
    RedisModule_SaveString(io, model->tag);
    RedisModule_SaveUnsigned(io, model->opts.batchsize);
    RedisModule_SaveUnsigned(io, model->opts.minbatchsize);
    RedisModule_SaveUnsigned(io, model->opts.minbatchtimeout);
    RedisModule_SaveUnsigned(io, model->ninputs);
    for (size_t i = 0; i < model->ninputs; i++) {
        RedisModule_SaveStringBuffer(io, model->inputs[i], strlen(model->inputs[i]) + 1);
    }
    RedisModule_SaveUnsigned(io, model->noutputs);
    for (size_t i = 0; i < model->noutputs; i++) {
        RedisModule_SaveStringBuffer(io, model->outputs[i], strlen(model->outputs[i]) + 1);
    }
    long long chunk_size = Config_GetModelChunkSize();
    const size_t n_chunks = len / chunk_size + 1;
    RedisModule_SaveUnsigned(io, len);
    RedisModule_SaveUnsigned(io, n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
        RedisModule_SaveStringBuffer(io, buffer + i * chunk_size, chunk_len);
    }

    if (buffer) {
        RedisModule_Free(buffer);
    }
}

void RAI_RDBSaveScript_v4(RedisModuleIO *io, void *value) {
    RAI_Script *script = (RAI_Script *)value;

    RedisModule_SaveStringBuffer(io, script->devicestr, strlen(script->devicestr) + 1);
    RedisModule_SaveString(io, script->tag);
    RedisModule_SaveStringBuffer(io, script->scriptdef, strlen(script->scriptdef) + 1);
    size_t nEntryPoints = array_len(script->entryPoints);

    RedisModule_SaveUnsigned(io, nEntryPoints);
    for (size_t i = 0; i < nEntryPoints; i++) {
        RedisModule_SaveStringBuffer(io, script->entryPoints[i],
                                     strlen(script->entryPoints[i]) + 1);
    }
}
