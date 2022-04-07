#include "decode_v0.h"
#include "assert.h"
#include "execution/run_queue_info.h"

void *RAI_RDBLoadTensor_v0(RedisModuleIO *io) {
    DLDevice device;
    device.device_type = RedisModule_LoadUnsigned(io);
    device.device_id = RedisModule_LoadUnsigned(io);

    // For now, we only support CPU tensors (except during model and script run)
    RedisModule_Assert(device.device_type == kDLCPU);
    if (device.device_id != -1) {
        device.device_id = -1;
    }

    DLDataType dtype;
    dtype.bits = RedisModule_LoadUnsigned(io);
    dtype.code = RedisModule_LoadUnsigned(io);
    dtype.lanes = RedisModule_LoadUnsigned(io);

    int ndims = RedisModule_LoadUnsigned(io);
    size_t shape[ndims];
    for (size_t i = 0; i < ndims; ++i) {
        shape[i] = RedisModule_LoadUnsigned(io);
    }

    int64_t strides[ndims];
    for (size_t i = 0; i < ndims; ++i) {
        strides[i] = RedisModule_LoadUnsigned(io);
    }
    size_t byte_offset = RedisModule_LoadUnsigned(io);

    size_t blob_len;
    char *data = RedisModule_LoadStringBuffer(io, &blob_len);
    if (RedisModule_IsIOError(io))
        goto error;

    RAI_Tensor *tensor = RAI_TensorNew(dtype, shape, ndims);
    tensor->blobSize = blob_len;
    tensor->tensor.dl_tensor.data = data;

    return tensor;

error:
    RedisModule_LogIOError(io, "error", "Experienced a short read while reading a tensor from RDB");
    return NULL;
}

void *RAI_RDBLoadModel_v0(RedisModuleIO *io) {
    char *devicestr = NULL;
    RedisModuleString *tag = NULL;
    size_t ninputs = 0;
    const char **inputs = NULL;
    size_t noutputs = 0;
    const char **outputs = NULL;
    char *buffer = NULL;

    RAI_Backend backend = RedisModule_LoadUnsigned(io);
    devicestr = RedisModule_LoadStringBuffer(io, NULL);
    size_t len;
    char *cstr_tag = RedisModule_LoadStringBuffer(io, &len);
    tag = RedisModule_CreateString(NULL, cstr_tag, len - 1);
    RedisModule_Free(cstr_tag);

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
        .backends_intra_op_parallelism = Config_GetBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = Config_GetBackendsInterOpParallelism(),
    };

    buffer = RedisModule_LoadStringBuffer(io, &len);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    RAI_Error err = {0};
    RAI_Model *model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs,
                                       outputs, buffer, len, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, backend);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "Could not load default backend");
            RAI_ClearError(&err);
            goto cleanup;
        }
        RAI_ClearError(&err);
        model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                buffer, len, &err);
    }

    if (err.code != RAI_OK) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        RedisModule_Log(ctx, "warning", "%s", err.detail);
        RAI_ClearError(&err);
        goto cleanup;
    }

    RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
    RedisModuleString *stats_keystr =
        RedisModule_CreateStringFromString(stats_ctx, RedisModule_GetKeyNameFromIO(io));

    RAI_RunStats *stats = RAI_StatsCreate(stats_keystr, RAI_MODEL, backend, devicestr, tag);
    RAI_StatsStoreEntry(stats_keystr, stats);
    model->info = stats;

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
    RedisModule_FreeString(NULL, stats_keystr);
    RedisModule_FreeString(NULL, tag);

    if (!RunQueue_IsExists(model->devicestr)) {
        RunQueue_Create(model->devicestr);
    }

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

    RedisModule_LogIOError(io, "warning",
                           "Experienced a short read while reading a model from RDB");
    return NULL;
}

void *RAI_RDBLoadScript_v0(RedisModuleIO *io) {
    RedisModuleString *tag = NULL;
    char *devicestr = NULL;
    char *scriptdef = NULL;
    RAI_Error err = {0};

    devicestr = RedisModule_LoadStringBuffer(io, NULL);
    size_t len;
    char *cstr_tag = RedisModule_LoadStringBuffer(io, &len);
    tag = RedisModule_CreateString(NULL, cstr_tag, len - 1);
    RedisModule_Free(cstr_tag);

    scriptdef = RedisModule_LoadStringBuffer(io, &len);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    RAI_Script *script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "Could not load default TORCH backend\n");
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

    RAI_RunStats *stats =
        RAI_StatsCreate(stats_keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);
    RAI_StatsStoreEntry(stats_keystr, stats);
    script->info = stats;

    RedisModule_FreeString(NULL, stats_keystr);
    RedisModule_FreeString(NULL, tag);
    RedisModule_Free(devicestr);
    RedisModule_Free(scriptdef);

    if (!RunQueue_IsExists(script->devicestr)) {
        RunQueue_Create(script->devicestr);
    }

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
