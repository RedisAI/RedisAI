/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v3.h"
#include "../v0/decode_v0.h"
#include "execution/run_queue_info.h"

/**
 * In case of IO errors, the default return values are:
 * numbers - 0
 * strings - null
 * So only when it is necessary check for IO errors.
 */

void *RAI_RDBLoadTensor_v3(RedisModuleIO *io) { return RAI_RDBLoadTensor_v0(io); }

void *RAI_RDBLoadModel_v3(RedisModuleIO *io) {

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
    const size_t minbatchtimeout = RedisModule_LoadUnsigned(io);

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
        .minbatchtimeout = minbatchtimeout,
        .backends_intra_op_parallelism = Config_GetBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = Config_GetBackendsInterOpParallelism(),
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
        RedisModule_FreeString(NULL, tag);
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

void *RAI_RDBLoadScript_v3(RedisModuleIO *io) {
    RedisModuleString *tag = NULL;
    char *devicestr = NULL;
    char *scriptdef = NULL;
    size_t nEntryPoints = 0;
    char **entryPoints = NULL;
    RAI_Error err = {0};

    size_t len;
    devicestr = RedisModule_LoadStringBuffer(io, &len);
    tag = RedisModule_LoadString(io);

    scriptdef = RedisModule_LoadStringBuffer(io, &len);
    if (RedisModule_IsIOError(io))
        goto cleanup;

    nEntryPoints = (size_t)RedisModule_LoadUnsigned(io);
    entryPoints = array_new(char *, nEntryPoints);

    for (size_t i = 0; i < nEntryPoints; i++) {
        char *entryPoint = RedisModule_LoadStringBuffer(io, &len);
        if (RedisModule_IsIOError(io)) {
            goto cleanup;
        }
        entryPoints = array_append(entryPoints, entryPoint);
    }

    RAI_Script *script = RAI_ScriptCompile(devicestr, tag, scriptdef, (const char **)entryPoints,
                                           nEntryPoints, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "Could not load default TORCH backend\n");
            RAI_ClearError(&err);
            goto cleanup;
        }
        RAI_ClearError(&err);
        script = RAI_ScriptCompile(devicestr, tag, scriptdef, (const char **)entryPoints,
                                   nEntryPoints, &err);
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
    for (size_t i = 0; i < nEntryPoints; i++) {
        RedisModule_Free(entryPoints[i]);
    }
    array_free(entryPoints);

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
    if (entryPoints) {
        for (size_t i = 0; i < nEntryPoints; i++) {
            RedisModule_Free(entryPoints[i]);
        }
        array_free(entryPoints);
    }
    return NULL;
}
