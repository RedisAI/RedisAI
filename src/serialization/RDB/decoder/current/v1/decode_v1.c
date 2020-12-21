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

    DLContext ctx;
    ctx.device_type = RedisModule_LoadUnsigned(io);
    ctx.device_id = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    // For now we only support CPU tensors (except during model and script run)
    assert(ctx.device_type == kDLCPU);
    assert(ctx.device_id == 0);

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

    RAI_Tensor *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->tensor = (DLManagedTensor){.dl_tensor = (DLTensor){.ctx = ctx,
                                                            .data = data,
                                                            .ndim = ndims,
                                                            .dtype = dtype,
                                                            .shape = shape,
                                                            .strides = strides,
                                                            .byte_offset = byte_offset},
                                    .manager_ctx = NULL,
                                    .deleter = NULL};
    ret->refCount = 1;
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
    const char **inputs = NULL;
    size_t noutputs = 0;
    const char **outputs = NULL;
    char *buffer = NULL;

    RAI_Backend backend = RedisModule_LoadUnsigned(io);
    devicestr = RedisModule_LoadStringBuffer(io, NULL);
    tag = RedisModule_LoadString(io);

    const size_t batchsize = RedisModule_LoadUnsigned(io);
    const size_t minbatchsize = RedisModule_LoadUnsigned(io);

    ninputs = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    inputs = RedisModule_Alloc(ninputs * sizeof(char *));

    for (size_t i = 0; i < ninputs; i++) {
        inputs[i] = RedisModule_LoadStringBuffer(io, NULL);
    }

    noutputs = RedisModule_LoadUnsigned(io);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    outputs = RedisModule_Alloc(noutputs * sizeof(char *));

    for (size_t i = 0; i < noutputs; i++) {
        outputs[i] = RedisModule_LoadStringBuffer(io, NULL);
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

    RAI_Error err = {0};
    RAI_Model *model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs,
                                       outputs, buffer, len, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, backend);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "error", "Could not load default backend");
            RAI_ClearError(&err);
            goto cleanup;
        }
        RAI_ClearError(&err);
        model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                buffer, len, &err);
    }

    if (err.code != RAI_OK) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        RedisModule_Log(ctx, "error", "%s", err.detail);
        RAI_ClearError(&err);
        goto cleanup;
    }

    RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
    RedisModuleString *stats_keystr =
        RedisModule_CreateStringFromString(stats_ctx, RedisModule_GetKeyNameFromIO(io));
    RedisModuleString *stats_tag = RAI_HoldString(NULL, tag);

    model->infokey =
        RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_MODEL, backend, devicestr, stats_tag);

    for (size_t i = 0; i < ninputs; i++) {
        RedisModule_Free((void *)inputs[i]);
    }
    RedisModule_Free(inputs);
    for (size_t i = 0; i < noutputs; i++) {
        RedisModule_Free((void *)outputs[i]);
    }
    RedisModule_Free(outputs);
    RedisModule_Free(buffer);

    RedisModule_Free(devicestr);
    RedisModule_Free(stats_keystr);

    return model;

cleanup:
    if (devicestr)
        RedisModule_Free(devicestr);
    if (tag)
        RedisModule_Free(tag);
    if (inputs) {
        for (size_t i = 0; i < ninputs; i++) {
            RedisModule_Free((void *)inputs[i]);
        }
        RedisModule_Free(inputs);
    }

    if (outputs) {
        for (size_t i = 0; i < noutputs; i++) {
            RedisModule_Free((void *)outputs[i]);
        }
        RedisModule_Free(outputs);
    }

    if (buffer)
        RedisModule_Free(buffer);

    RedisModule_LogIOError(io, "error", "Experienced a short read while reading a model from RDB");
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

    const char *stats_devicestr = RedisModule_Strdup(devicestr);

    tag = RAI_HoldString(NULL, tag);

    script->infokey =
        RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

    RedisModule_FreeString(NULL, stats_keystr);

    return script;
cleanup:
    if (devicestr)
        RedisModule_Free(devicestr);
    if (scriptdef)
        RedisModule_Free(scriptdef);
    return NULL;
}
