#include "decode_v1.h"
#include "assert.h"

/**
 * In case of IO errors, the default return values are:
 * numbers - 0
 * strings - null
 * So only when it is necessary check for IO errors.
 */

void *RAI_RDBLoadTensor_v1(RedisModuleIO *io) {
    int64_t *shape = NULL;
    int64_t *strides = NULL;

    DLDevice device;
    device.device_type = RedisModule_LoadUnsigned(io);
    device.device_id = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    // For now we only support CPU tensors (except during model and script run)
    assert(device.device_type == kDLCPU);
    assert(device.device_id == 0);

    DLDataType dtype;
    dtype.bits = RedisModule_LoadUnsigned(io);
    dtype.code = RedisModule_LoadUnsigned(io);
    dtype.lanes = RedisModule_LoadUnsigned(io);

    size_t ndims = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    shape = RedisModule_Calloc(ndims, sizeof(*shape));
    for (size_t i = 0; i < ndims; ++i) {
        shape[i] = RedisModule_LoadUnsigned(io);
    }

    strides = RedisModule_Calloc(ndims, sizeof(*strides));
    for (size_t i = 0; i < ndims; ++i) {
        strides[i] = RedisModule_LoadUnsigned(io);
    }

    size_t byte_offset = RedisModule_LoadUnsigned(io);

    size_t len;
    char *data = RedisModule_LoadStringBuffer(io, &len);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    RAI_Tensor *ret = RAI_TensorNew();
    ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.device = device,
                                                            .data = data,
                                                            .ndim = ndims,
                                                            .dtype = dtype,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = byte_offset},
                                    .manager_ctx = NULL,
                                    .deleter = NULL};
    return ret;

cleanup:
    if (shape)
        RedisModule_Free(shape);
    if (strides)
        RedisModule_Free(strides);
    RedisModule_LogIOError(io, "error", "Experienced a short read while reading a tensor from RDB");
    return NULL;
}

void *RAI_RDBLoadModel_v1(RedisModuleIO *io) {

    char *devicestr = NULL;
    RedisModuleString *tag = NULL;
    size_t ninputs = 0;
    char **inputs = NULL;
    size_t noutputs = 0;
    char **outputs = NULL;
    char *buffer = NULL;
    RAI_Error err = {0};
    char *error_str = "Experienced a short read while reading a model from RDB";

    RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
    RedisModuleString *key_str =
        RedisModule_CreateStringFromString(NULL, RedisModule_GetKeyNameFromIO(io));
    if (!key_str) {
        RedisModule_LogIOError(io, "error", "Couldn't get model key name from RDB");
        return NULL;
    }
    RAI_Backend backend = RedisModule_LoadUnsigned(io);
    devicestr = RedisModule_LoadStringBuffer(io, NULL);
    tag = RedisModule_LoadString(io);
    const size_t batchsize = RedisModule_LoadUnsigned(io);
    const size_t minbatchsize = RedisModule_LoadUnsigned(io);

    ninputs = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;
    inputs = array_new(char *, ninputs);
    for (size_t i = 0; i < ninputs; i++) {
        inputs = array_append(inputs, RedisModule_LoadStringBuffer(io, NULL));
    }

    noutputs = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;
    outputs = array_new(char *, noutputs);
    for (size_t i = 0; i < noutputs; i++) {
        outputs = array_append(outputs, RedisModule_LoadStringBuffer(io, NULL));
    }

    RAI_ModelOpts opts = {
        .batchsize = batchsize,
        .minbatchsize = minbatchsize,
        .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
    };

    size_t len = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    buffer = RedisModule_Alloc(len);
    const size_t n_chunks = RedisModule_LoadUnsigned(io);
    long long chunk_offset = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_len;
        char *chunk_buffer = RedisModule_LoadStringBuffer(io, &chunk_len);
        if (RedisModule_IsIOError(io))
            goto cleanup;
        memcpy(buffer + chunk_offset, chunk_buffer, chunk_len);
        chunk_offset += chunk_len;
        RedisModule_Free(chunk_buffer);
    }

    RAI_Model *model = RedisModule_Calloc(1, sizeof(*model));
    model->infokey = RAI_HoldString(NULL, key_str);
    model->backend = backend;
    model->devicestr = devicestr;
    model->tag = tag;
    model->inputs = inputs;
    model->ninputs = ninputs;
    model->outputs = outputs;
    model->noutputs = noutputs;
    model->opts = opts;
    model->data = buffer;
    model->datalen = len;

    const char *backend_str = RAI_BackendName(backend);
    if (ModelCreateBE(model, &err) != REDISMODULE_OK) {
        // If we got an error *not* because of lazy loading, we fail and unblock.
        if (RAI_GetErrorCode(&err) != RAI_EBACKENDNOTLOADED) {
            error_str = (char *)RAI_GetError(&err);
            goto cleanup;
        }
        RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend",
                        backend_str);
        int ret = RAI_LoadDefaultBackend(NULL, model->backend);
        if (ret != REDISMODULE_OK) {
            sprintf(error_str, "could not load %s default backend", backend_str);
            goto cleanup;
        }
        // Try creating model for backend again.
        RAI_ClearError(&err);
        if (ModelCreateBE(model, &err) != REDISMODULE_OK) {
            error_str = (char *)RAI_GetError(&err);
            goto cleanup;
        }
    }
    RAI_AddStatsEntry(ctx, key_str, RAI_MODEL, backend, devicestr, tag);

    return model;

cleanup:
    if (devicestr)
        RedisModule_Free(devicestr);
    if (tag)
        RedisModule_FreeString(NULL, tag);
    if (inputs) {
        for (size_t i = 0; i < ninputs; i++) {
            RedisModule_Free(inputs[i]);
        }
        array_free(inputs);
    }

    if (outputs) {
        for (size_t i = 0; i < noutputs; i++) {
            RedisModule_Free(outputs[i]);
        }
        array_free(outputs);
    }

    if (buffer)
        RedisModule_Free(buffer);

    RedisModule_LogIOError(io, "error", "%s", error_str);
    if (RAI_GetErrorCode(&err) != RAI_OK) {
        RAI_ClearError(&err);
    }
    return NULL;
}

void *RAI_RDBLoadScript_v1(RedisModuleIO *io) {
    RedisModuleString *tag = NULL;
    char *devicestr = NULL;
    char *scriptdef = NULL;
    RAI_Error err = {0};

    devicestr = RedisModule_LoadStringBuffer(io, NULL);
    tag = RedisModule_LoadString(io);

    size_t len;
    scriptdef = RedisModule_LoadStringBuffer(io, &len);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    RAI_Script *script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "error", "Could not load default TORCH backend\n");
            RAI_ClearError(&err);
            goto cleanup;
        }
        RAI_ClearError(&err);
        script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
    }

    if (err.code != RAI_OK) {
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
        goto cleanup;
    }

    RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
    RedisModuleString *stats_keystr =
        RedisModule_CreateStringFromString(stats_ctx, RedisModule_GetKeyNameFromIO(io));

    script->infokey =
        RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

    RedisModule_FreeString(NULL, stats_keystr);
    RedisModule_FreeString(NULL, tag);
    RedisModule_Free(devicestr);
    RedisModule_Free(scriptdef);
    return script;
cleanup:
    if (devicestr)
        RedisModule_Free(devicestr);
    if (scriptdef)
        RedisModule_Free(scriptdef);
    if (tag)
        RedisModule_FreeString(NULL, tag);
    return NULL;
}
