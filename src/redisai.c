
#ifdef __linux__
#define _GNU_SOURCE
#endif

#define REDISMODULE_MAIN
#include "redismodule.h"
#include "tensor.h"
#include "command_parser.h"
#include "backends.h"
#include "backends/util.h"
#include "background_workers.h"
#include "DAG/dag.h"
#include "DAG/dag_builder.h"
#include "DAG/dag_execute.h"
#include "model.h"
#include "modelRun_ctx.h"
#include "script.h"
#include "stats.h"
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include "util/queue.h"
#include "version.h"

#include "redis_ai_types/model_type.h"
#include "redis_ai_types/script_type.h"
#include "redis_ai_types/tensor_type.h"

#define REDISAI_H_INCLUDE
#include "redisai.h"
#undef REDISAI_H_INCLUDE

#ifndef REDISAI_GIT_SHA
#define REDISAI_GIT_SHA "unknown"
#endif

#ifdef __linux__
#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 1
#endif
#endif

int redisMajorVersion;
int redisMinorVersion;
int redisPatchVersion;

int rlecMajorVersion;
int rlecMinorVersion;
int rlecPatchVersion;
int rlecBuild;

void getRedisVersion() {
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RedisModuleCallReply *reply = RedisModule_Call(ctx, "info", "c", "server");
    assert(RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING);
    size_t len;
    const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);

    int n = sscanf(replyStr, "# Server\nredis_version:%d.%d.%d", &redisMajorVersion,
                   &redisMinorVersion, &redisPatchVersion);

    assert(n == 3);

    rlecMajorVersion = -1;
    rlecMinorVersion = -1;
    rlecPatchVersion = -1;
    rlecBuild = -1;
    char *enterpriseStr = strstr(replyStr, "rlec_version:");
    if (enterpriseStr) {
        n = sscanf(enterpriseStr, "rlec_version:%d.%d.%d-%d", &rlecMajorVersion, &rlecMinorVersion,
                   &rlecPatchVersion, &rlecBuild);
        if (n != 4) {
            RedisModule_Log(NULL, "warning", "Could not extract enterprise version");
        }
    }

    RedisModule_FreeCallReply(reply);
    RedisModule_FreeThreadSafeContext(ctx);
}

static inline int IsEnterprise() { return rlecMajorVersion != -1; }

/* ----------------------- RedisAI Module Commands ------------------------- */

/**
 * AI.TENSORSET key type dim1..dimN [BLOB data | VALUES val1..valN]
 */
int RedisAI_TensorSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 4)
        return RedisModule_WrongArity(ctx);

    RedisModuleKey *key;
    RAI_Error err = {0};
    const int status =
        RAI_OpenKey_Tensor(ctx, argv[1], &key, REDISMODULE_READ | REDISMODULE_WRITE, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }

    RAI_Tensor *t = NULL;
    const int parse_result = RAI_parseTensorSetArgs(argv, argc, &t, 1, &err);

    // if the number of parsed args is negative something went wrong
    if (parse_result < 0) {
        RedisModule_CloseKey(key);
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }

    if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t) != REDISMODULE_OK) {
        RAI_TensorFree(t);
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
    }
    RedisModule_CloseKey(key);
    RedisModule_ReplyWithSimpleString(ctx, "OK");
    RedisModule_ReplicateVerbatim(ctx);
    return REDISMODULE_OK;
}

/**
 * AI.TENSORGET tensor_key [META] [BLOB | VALUES]
 */
int RedisAI_TensorGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 2 || argc > 4)
        return RedisModule_WrongArity(ctx);

    RAI_Tensor *t;
    RedisModuleKey *key;
    RAI_Error err = {0};
    const int status = RAI_GetTensorFromKeyspace(ctx, argv[1], &key, &t, REDISMODULE_READ, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }
    uint fmt = ParseTensorGetArgs(&err, argv, argc);
    if (fmt == TENSOR_NONE) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        // This means that args are invalid.
        return REDISMODULE_ERR;
    }
    ReplyWithTensor(ctx, fmt, t);
    return REDISMODULE_OK;
}

/**
 * AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]]
 * [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] BLOB model_blob
 */
int RedisAI_ModelSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 4)
        return RedisModule_WrongArity(ctx);

    ArgsCursor ac;
    ArgsCursor_InitRString(&ac, argv + 1, argc - 1);

    RedisModuleString *keystr;
    AC_GetRString(&ac, &keystr, 0);

    const char *bckstr;
    int backend;
    AC_GetString(&ac, &bckstr, NULL, 0);
    if (strcasecmp(bckstr, "TF") == 0) {
        backend = RAI_BACKEND_TENSORFLOW;
    } else if (strcasecmp(bckstr, "TFLITE") == 0) {
        backend = RAI_BACKEND_TFLITE;
    } else if (strcasecmp(bckstr, "TORCH") == 0) {
        backend = RAI_BACKEND_TORCH;
    } else if (strcasecmp(bckstr, "ONNX") == 0) {
        backend = RAI_BACKEND_ONNXRUNTIME;
    } else {
        return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
    }

    const char *devicestr;
    AC_GetString(&ac, &devicestr, NULL, 0);

    if (strlen(devicestr) > 10 || strcasecmp(devicestr, "INPUTS") == 0 ||
        strcasecmp(devicestr, "OUTPUTS") == 0 || strcasecmp(devicestr, "TAG") == 0 ||
        strcasecmp(devicestr, "BATCHSIZE") == 0 || strcasecmp(devicestr, "MINBATCHSIZE") == 0 ||
        strcasecmp(devicestr, "MINBATCHTIMEOUT") == 0 || strcasecmp(devicestr, "BLOB") == 0) {
        return RedisModule_ReplyWithError(ctx, "ERR Invalid DEVICE");
    }

    RedisModuleString *tag = NULL;
    if (AC_AdvanceIfMatch(&ac, "TAG")) {
        AC_GetRString(&ac, &tag, 0);
    }

    unsigned long long batchsize = 0;
    if (AC_AdvanceIfMatch(&ac, "BATCHSIZE")) {
        if (backend == RAI_BACKEND_TFLITE) {
            return RedisModule_ReplyWithError(
                ctx, "ERR Auto-batching not supported by the TFLITE backend");
        }
        if (AC_GetUnsignedLongLong(&ac, &batchsize, 0) != AC_OK) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for BATCHSIZE");
        }
    }

    unsigned long long minbatchsize = 0;
    if (AC_AdvanceIfMatch(&ac, "MINBATCHSIZE")) {
        if (batchsize == 0) {
            return RedisModule_ReplyWithError(ctx, "ERR MINBATCHSIZE specified without BATCHSIZE");
        }
        if (AC_GetUnsignedLongLong(&ac, &minbatchsize, 0) != AC_OK) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHSIZE");
        }
    }

    unsigned long long minbatchtimeout = 0;
    if (AC_AdvanceIfMatch(&ac, "MINBATCHTIMEOUT")) {
        if (batchsize == 0) {
            return RedisModule_ReplyWithError(ctx,
                                              "ERR MINBATCHTIMEOUT specified without BATCHSIZE");
        }
        if (minbatchsize == 0) {
            return RedisModule_ReplyWithError(ctx,
                                              "ERR MINBATCHTIMEOUT specified without MINBATCHSIZE");
        }
        if (AC_GetUnsignedLongLong(&ac, &minbatchtimeout, 0) != AC_OK) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHTIMEOUT");
        }
    }

    if (AC_IsAtEnd(&ac)) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
    }

    ArgsCursor optionsac;
    const char *blob_matches[] = {"BLOB"};
    AC_GetSliceUntilMatches(&ac, &optionsac, 1, blob_matches);

    if (optionsac.argc == 0 && backend == RAI_BACKEND_TENSORFLOW) {
        return RedisModule_ReplyWithError(
            ctx, "ERR Insufficient arguments, INPUTS and OUTPUTS not specified");
    }

    ArgsCursor inac = {0};
    ArgsCursor outac = {0};
    if (optionsac.argc > 0 && backend == RAI_BACKEND_TENSORFLOW) {
        if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
            return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified");
        }

        const char *matches[] = {"OUTPUTS"};
        AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);

        if (!AC_IsAtEnd(&optionsac)) {
            if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
                return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified");
            }

            AC_GetSliceToEnd(&optionsac, &outac);
        }
    }

    size_t ninputs = inac.argc;
    const char *inputs[ninputs];
    for (size_t i = 0; i < ninputs; i++) {
        AC_GetString(&inac, inputs + i, NULL, 0);
    }

    size_t noutputs = outac.argc;
    const char *outputs[noutputs];
    for (size_t i = 0; i < noutputs; i++) {
        AC_GetString(&outac, outputs + i, NULL, 0);
    }

    RAI_ModelOpts opts = {
        .batchsize = batchsize,
        .minbatchsize = minbatchsize,
        .minbatchtimeout = minbatchtimeout,
        .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
    };

    RAI_Model *model = NULL;

    AC_AdvanceUntilMatches(&ac, 1, blob_matches);

    if (AC_IsAtEnd(&ac)) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
    }

    AC_Advance(&ac);

    ArgsCursor blobsac;
    AC_GetSliceToEnd(&ac, &blobsac);

    size_t modellen;
    char *modeldef;

    if (blobsac.argc == 1) {
        AC_GetString(&blobsac, (const char **)&modeldef, &modellen, 0);
    } else {
        const char *chunks[blobsac.argc];
        size_t chunklens[blobsac.argc];
        modellen = 0;
        while (!AC_IsAtEnd(&blobsac)) {
            AC_GetString(&blobsac, &chunks[blobsac.offset], &chunklens[blobsac.offset], 0);
            modellen += chunklens[blobsac.offset - 1];
        }

        modeldef = RedisModule_Calloc(modellen, sizeof(char));
        size_t offset = 0;
        for (size_t i = 0; i < blobsac.argc; i++) {
            memcpy(modeldef + offset, chunks[i], chunklens[i]);
            offset += chunklens[i];
        }
    }

    RAI_Error err = {0};

    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                            modeldef, modellen, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend",
                        bckstr);
        int ret = RAI_LoadDefaultBackend(ctx, backend);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "error", "could not load %s default backend", bckstr);
            int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
            RAI_ClearError(&err);
            return ret;
        }
        RAI_ClearError(&err);
        model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                modeldef, modellen, &err);
    }

    if (blobsac.argc > 1) {
        RedisModule_Free(modeldef);
    }

    if (err.code != RAI_OK) {
        RedisModule_Log(ctx, "error", "%s", err.detail);
        int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
        RAI_ClearError(&err);
        return ret;
    }

    // TODO: if backend loaded, make sure there's a queue
    RunQueueInfo *run_queue_info = NULL;
    if (ensureRunQueue(devicestr, &run_queue_info) != REDISMODULE_OK) {
        RAI_ModelFree(model, &err);
        if (err.code != RAI_OK) {
            RedisModule_Log(ctx, "error", "%s", err.detail);
            int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
            RAI_ClearError(&err);
            return ret;
        }
        return RedisModule_ReplyWithError(ctx,
                                          "ERR Could not initialize queue on requested device");
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ | REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
        RedisModule_CloseKey(key);
        RAI_ModelFree(model, &err);
        if (err.code != RAI_OK) {
            RedisModule_Log(ctx, "error", "%s", err.detail);
            int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
            RAI_ClearError(&err);
            return ret;
        }
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }

    RedisModule_ModuleTypeSetValue(key, RedisAI_ModelType, model);

    model->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_MODEL, backend, devicestr, tag);

    RedisModule_CloseKey(key);

    RedisModule_ReplyWithSimpleString(ctx, "OK");

    RedisModule_ReplicateVerbatim(ctx);

    return REDISMODULE_OK;
}

void RAI_ReplyWithChunks(RedisModuleCtx *ctx, const char *buffer, long long len) {
    long long chunk_size = getModelChunkSize();
    const size_t n_chunks = len / chunk_size + 1;
    if (n_chunks > 1) {
        RedisModule_ReplyWithArray(ctx, (long)n_chunks);
        for (size_t i = 0; i < n_chunks; i++) {
            size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
            RedisModule_ReplyWithStringBuffer(ctx, buffer + i * chunk_size, chunk_len);
        }
    } else {
        RedisModule_ReplyWithStringBuffer(ctx, buffer, len);
    }
}

/**
 * AI.MODELGET model_key [META] [BLOB]
 */
int RedisAI_ModelGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 2 || argc > 4)
        return RedisModule_WrongArity(ctx);

    RAI_Error err = {0};
    RAI_Model *mto;
    const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &mto, REDISMODULE_READ, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }

    int meta = 0;
    int blob = 0;
    for (int i = 2; i < argc; i++) {
        const char *optstr = RedisModule_StringPtrLen(argv[i], NULL);
        if (!strcasecmp(optstr, "META")) {
            meta = 1;
        } else if (!strcasecmp(optstr, "BLOB")) {
            blob = 1;
        }
    }

    if (!meta && !blob) {
        return RedisModule_ReplyWithError(ctx, "ERR no META or BLOB specified");
    }

    char *buffer = NULL;
    size_t len = 0;

    if (blob) {
        RAI_ModelSerialize(mto, &buffer, &len, &err);
        if (err.code != RAI_OK) {
#ifdef RAI_PRINT_BACKEND_ERRORS
            printf("ERR: %s\n", err.detail);
#endif
            int ret = RedisModule_ReplyWithError(ctx, err.detail);
            RAI_ClearError(&err);
            if (*buffer) {
                RedisModule_Free(buffer);
            }
            return ret;
        }
    }

    if (!meta && blob) {
        RAI_ReplyWithChunks(ctx, buffer, len);
        RedisModule_Free(buffer);
        return REDISMODULE_OK;
    }

    const int outentries = blob ? 16 : 14;
    RedisModule_ReplyWithArray(ctx, outentries);

    RedisModule_ReplyWithCString(ctx, "backend");
    const char *backendstr = RAI_BackendName(mto->backend);
    RedisModule_ReplyWithCString(ctx, backendstr);

    RedisModule_ReplyWithCString(ctx, "device");
    RedisModule_ReplyWithCString(ctx, mto->devicestr);

    RedisModule_ReplyWithCString(ctx, "tag");
    RedisModuleString *empty_tag = RedisModule_CreateString(NULL, "", 0);
    RedisModule_ReplyWithString(ctx, mto->tag ? mto->tag : empty_tag);
    RedisModule_FreeString(NULL, empty_tag);

    RedisModule_ReplyWithCString(ctx, "batchsize");
    RedisModule_ReplyWithLongLong(ctx, (long)mto->opts.batchsize);

    RedisModule_ReplyWithCString(ctx, "minbatchsize");
    RedisModule_ReplyWithLongLong(ctx, (long)mto->opts.minbatchsize);

    RedisModule_ReplyWithCString(ctx, "inputs");
    const size_t ninputs = array_len(mto->inputs);
    RedisModule_ReplyWithArray(ctx, (long)ninputs);

    for (size_t i = 0; i < ninputs; i++) {
        RedisModule_ReplyWithCString(ctx, mto->inputs[i]);
    }

    RedisModule_ReplyWithCString(ctx, "outputs");
    const size_t noutputs = array_len(mto->outputs);
    RedisModule_ReplyWithArray(ctx, (long)noutputs);

    for (size_t i = 0; i < noutputs; i++) {
        RedisModule_ReplyWithCString(ctx, mto->outputs[i]);
    }

    if (meta && blob) {
        RedisModule_ReplyWithCString(ctx, "blob");
        RAI_ReplyWithChunks(ctx, buffer, len);
        RedisModule_Free(buffer);
    }

    return REDISMODULE_OK;
}

/**
 * AI.MODELDEL model_key
 */
int RedisAI_ModelDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 2)
        return RedisModule_WrongArity(ctx);

    RAI_Model *mto;
    RAI_Error err = {0};
    const int status =
        RAI_GetModelFromKeyspace(ctx, argv[1], &mto, REDISMODULE_READ | REDISMODULE_WRITE, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_WRITE);
    RedisModule_DeleteKey(key);
    RedisModule_CloseKey(key);
    RedisModule_ReplicateVerbatim(ctx);

    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/**
 * AI._MODELSCAN
 */
int RedisAI_ModelScan_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 1)
        return RedisModule_WrongArity(ctx);

    RedisModule_Log(ctx, "warning",
                    "MODELSCAN is experimental and might be removed in future versions");

    long long nkeys;
    RedisModuleString **keys;
    RedisModuleString **tags;
    RAI_ListStatsEntries(RAI_MODEL, &nkeys, &keys, &tags);

    RedisModule_ReplyWithArray(ctx, nkeys);

    RedisModuleString *empty_tag = RedisModule_CreateString(NULL, "", 0);
    for (long long i = 0; i < nkeys; i++) {
        RedisModule_ReplyWithArray(ctx, 2);
        RedisModule_ReplyWithString(ctx, keys[i]);
        RedisModule_ReplyWithString(ctx, tags[i] ? tags[i] : empty_tag);
    }
    RedisModule_FreeString(NULL, empty_tag);

    RedisModule_Free(keys);
    RedisModule_Free(tags);

    return REDISMODULE_OK;
}

/**
 * AI.MODELRUN model_key [TIMEOUT t] INPUTS <input_key> [input_key ...] OUTPUTS
 * <output_key> [output_key ...]
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_MODELRUN, false);
}

/**
 * AI.SCRIPTRUN <key> <function> INPUTS <input_key> [input_key ...] OUTPUTS
 * <output_key> [output_key ...]
 */
int RedisAI_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_ScriptRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }

    // Convert The script run command into a DAG command that contains a single op.
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_SCRIPTRUN, false);
}

/**
 * AI.SCRIPTGET script_key [META] [SOURCE]
 */
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 2 || argc > 4)
        return RedisModule_WrongArity(ctx);

    RAI_Script *sto;
    RAI_Error err = {0};
    const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &sto, REDISMODULE_READ, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }

    int meta = 0;
    int source = 0;
    for (int i = 2; i < argc; i++) {
        const char *optstr = RedisModule_StringPtrLen(argv[i], NULL);
        if (!strcasecmp(optstr, "META")) {
            meta = 1;
        } else if (!strcasecmp(optstr, "SOURCE")) {
            source = 1;
        }
    }

    if (!meta && !source) {
        return RedisModule_ReplyWithError(ctx, "ERR no META or SOURCE specified");
    }

    if (!meta && source) {
        RedisModule_ReplyWithCString(ctx, sto->scriptdef);
        return REDISMODULE_OK;
    }

    int outentries = source ? 6 : 4;

    RedisModule_ReplyWithArray(ctx, outentries);
    RedisModule_ReplyWithCString(ctx, "device");
    RedisModule_ReplyWithCString(ctx, sto->devicestr);
    RedisModule_ReplyWithCString(ctx, "tag");
    RedisModule_ReplyWithString(ctx, sto->tag);
    if (source) {
        RedisModule_ReplyWithCString(ctx, "source");
        RedisModule_ReplyWithCString(ctx, sto->scriptdef);
    }
    return REDISMODULE_OK;
}

/**
 * AI.SCRIPTDEL script_key
 */
int RedisAI_ScriptDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 2)
        return RedisModule_WrongArity(ctx);

    RAI_Script *sto;
    RAI_Error err = {0};
    const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &sto, REDISMODULE_WRITE, &err);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&err));
        RAI_ClearError(&err);
        return REDISMODULE_ERR;
    }
    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_WRITE);
    RedisModule_DeleteKey(key);
    RedisModule_CloseKey(key);

    RedisModule_ReplicateVerbatim(ctx);
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/**
 * AI.SCRIPTSET script_key device [TAG tag] SOURCE script_source
 */
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 5 && argc != 7)
        return RedisModule_WrongArity(ctx);

    ArgsCursor ac;
    ArgsCursor_InitRString(&ac, argv + 1, argc - 1);

    RedisModuleString *keystr;
    AC_GetRString(&ac, &keystr, 0);

    const char *devicestr;
    AC_GetString(&ac, &devicestr, NULL, 0);

    RedisModuleString *tag = NULL;
    if (AC_AdvanceIfMatch(&ac, "TAG")) {
        AC_GetRString(&ac, &tag, 0);
    }

    if (AC_IsAtEnd(&ac)) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing script SOURCE");
    }

    size_t scriptlen;
    const char *scriptdef = NULL;

    if (AC_AdvanceIfMatch(&ac, "SOURCE")) {
        AC_GetString(&ac, &scriptdef, &scriptlen, 0);
    }

    if (scriptdef == NULL) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing script SOURCE");
    }

    RAI_Script *script = NULL;

    RAI_Error err = {0};
    script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModule_Log(ctx, "warning",
                        "Backend TORCH not loaded, will try loading default backend");
        int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "error", "Could not load TORCH default backend");
            int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
            RAI_ClearError(&err);
            return ret;
        }
        RAI_ClearError(&err);
        script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
    }

    if (err.code != RAI_OK) {
#ifdef RAI_PRINT_BACKEND_ERRORS
        printf("ERR: %s\n", err.detail);
#endif
        int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
        RAI_ClearError(&err);
        return ret;
    }

    RunQueueInfo *run_queue_info = NULL;
    // If the queue does not exist, initialize it
    if (ensureRunQueue(devicestr, &run_queue_info) == REDISMODULE_ERR) {
        RAI_ScriptFree(script, &err);
        if (err.code != RAI_OK) {
#ifdef RAI_PRINT_BACKEND_ERRORS
            printf("ERR: %s\n", err.detail);
#endif
            int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
            RAI_ClearError(&err);
            return ret;
        }
        return RedisModule_ReplyWithError(ctx,
                                          "ERR Could not initialize queue on requested device");
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ | REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }

    RedisModule_ModuleTypeSetValue(key, RedisAI_ScriptType, script);

    script->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

    RedisModule_CloseKey(key);

    RedisModule_ReplyWithSimpleString(ctx, "OK");

    RedisModule_ReplicateVerbatim(ctx);

    return REDISMODULE_OK;
}

/**
 * AI._SCRIPTSCAN
 */
int RedisAI_ScriptScan_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 1)
        return RedisModule_WrongArity(ctx);

    RedisModule_Log(ctx, "warning",
                    "SCRIPTSCAN is experimental and might be removed in future versions");

    long long nkeys;
    RedisModuleString **keys;
    RedisModuleString **tags;
    RAI_ListStatsEntries(RAI_SCRIPT, &nkeys, &keys, &tags);

    RedisModule_ReplyWithArray(ctx, nkeys);

    RedisModuleString *empty_tag = RedisModule_CreateString(NULL, "", 0);
    for (long long i = 0; i < nkeys; i++) {
        RedisModule_ReplyWithArray(ctx, 2);
        RedisModule_ReplyWithString(ctx, keys[i]);
        RedisModule_ReplyWithString(ctx, tags[i] ? tags[i] : empty_tag);
    }
    RedisModule_FreeString(NULL, empty_tag);

    RedisModule_Free(keys);
    RedisModule_Free(tags);

    return REDISMODULE_OK;
}

/**
 * AI.INFO <model_or_script_key> [RESETSTAT]
 */
int RedisAI_Info_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 2 && argc != 3)
        return RedisModule_WrongArity(ctx);
    RedisModuleString *runkey = argv[1];
    struct RedisAI_RunStats *rstats = NULL;
    if (RAI_GetRunStats(runkey, &rstats) == REDISMODULE_ERR) {
        return RedisModule_ReplyWithError(ctx, "ERR cannot find run info for key");
    }

    if (argc == 3) {
        const char *subcommand = RedisModule_StringPtrLen(argv[2], NULL);
        if (!strcasecmp(subcommand, "RESETSTAT")) {
            RAI_ResetRunStats(rstats);
            RedisModule_ReplyWithSimpleString(ctx, "OK");
            return REDISMODULE_OK;
        }
    }

    RedisModule_ReplyWithArray(ctx, 18);

    RedisModule_ReplyWithCString(ctx, "key");
    RedisModule_ReplyWithString(ctx, rstats->key);
    RedisModule_ReplyWithCString(ctx, "type");
    if (rstats->type == 0) {
        RedisModule_ReplyWithCString(ctx, "MODEL");
    } else {
        RedisModule_ReplyWithCString(ctx, "SCRIPT");
    }
    RedisModule_ReplyWithCString(ctx, "backend");
    RedisModule_ReplyWithCString(ctx, RAI_BackendName(rstats->backend));
    RedisModule_ReplyWithCString(ctx, "device");
    RedisModule_ReplyWithCString(ctx, rstats->devicestr);
    RedisModule_ReplyWithCString(ctx, "tag");
    if (rstats->tag) {
        RedisModule_ReplyWithString(ctx, rstats->tag);
    } else {
        RedisModule_ReplyWithCString(ctx, "");
    }
    RedisModule_ReplyWithCString(ctx, "duration");
    RedisModule_ReplyWithLongLong(ctx, rstats->duration_us);
    RedisModule_ReplyWithCString(ctx, "samples");
    if (rstats->type == 0) {
        RedisModule_ReplyWithLongLong(ctx, rstats->samples);
    } else {
        RedisModule_ReplyWithLongLong(ctx, -1);
    }
    RedisModule_ReplyWithCString(ctx, "calls");
    RedisModule_ReplyWithLongLong(ctx, rstats->calls);
    RedisModule_ReplyWithCString(ctx, "errors");
    RedisModule_ReplyWithLongLong(ctx, rstats->nerrors);

    return REDISMODULE_OK;
}

/**
* AI.CONFIG [BACKENDSPATH <default_location_of_backend_libraries> |
             LOADBACKEND <backend_identifier> <location_of_backend_library> |
             CHUNKLEN <len>]
*/
int RedisAI_Config_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 2)
        return RedisModule_WrongArity(ctx);

    const char *subcommand = RedisModule_StringPtrLen(argv[1], NULL);
    if (!strcasecmp(subcommand, "LOADBACKEND")) {
        return RedisAI_Config_LoadBackend(ctx, argv + 1, argc - 1);
    }

    if (!strcasecmp(subcommand, "BACKENDSPATH")) {
        if (argc > 2) {
            return RedisAI_Config_BackendsPath(ctx, RedisModule_StringPtrLen(argv[2], NULL));
        } else {
            return RedisModule_ReplyWithError(ctx, "ERR BACKENDSPATH: missing path argument");
        }
    }

    if (!strcasecmp(subcommand, "MODEL_CHUNK_SIZE")) {
        if (argc > 2) {
            RedisAI_Config_ModelChunkSize(argv[2]);
            return RedisModule_ReplyWithSimpleString(ctx, "OK");
        } else {
            return RedisModule_ReplyWithError(ctx, "ERR MODEL_CHUNK_SIZE: missing chunk size");
        }
    }

    return RedisModule_ReplyWithError(ctx, "ERR unsupported subcommand");
}

/**
 * AI.DAGRUN [LOAD <nkeys> key1 key2... ] [PERSIST <nkeys> key1 key2... ]
 * [TIMEOUT t] |> [COMMAND1] |> [COMMAND2] |> [COMMANDN]
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGRUN, false);
}

/**
 * AI.DAGRUN_RO [LOAD <nkeys> key1 key2... ] [TIMEOUT t] |> [COMMAND1] |>
 * [COMMAND2] |> [COMMANDN]
 *
 * Read-only (no PERSIST) DAG execution.
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagRunRO_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGRUN, true);
}

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, ctx)                                                                    \
    if (RedisModule_ExportSharedAPI) {                                                             \
        if (RedisModule_ExportSharedAPI(ctx, "RedisAI_" #name, RAI_##name) != REDISMODULE_OK) {    \
            RedisModule_Log(ctx, "warning", "Could not register RedisAI_%s", #name);               \
            return REDISMODULE_ERR;                                                                \
        }                                                                                          \
    }

static int RAI_GetLLAPIVersion() { return REDISAI_LLAPI_VERSION; }

static int RedisAI_RegisterApi(RedisModuleCtx *ctx) {

    if (!RedisModule_ExportSharedAPI) {
        RedisModule_Log(ctx, "warning",
                        "Redis version does not support SharedAPI; running without "
                        "exposing C API to other modules");
    }

    REGISTER_API(GetLLAPIVersion, ctx);

    REGISTER_API(InitError, ctx);
    REGISTER_API(ClearError, ctx);
    REGISTER_API(FreeError, ctx);
    REGISTER_API(GetError, ctx);
    REGISTER_API(GetErrorOneLine, ctx);
    REGISTER_API(GetErrorCode, ctx);
    REGISTER_API(CloneError, ctx);

    REGISTER_API(TensorCreate, ctx);
    REGISTER_API(TensorCreateByConcatenatingTensors, ctx);
    REGISTER_API(TensorCreateBySlicingTensor, ctx);
    REGISTER_API(TensorLength, ctx);
    REGISTER_API(TensorDataSize, ctx);
    REGISTER_API(TensorFree, ctx);
    REGISTER_API(TensorSetData, ctx);
    REGISTER_API(TensorSetValueFromLongLong, ctx);
    REGISTER_API(TensorSetValueFromDouble, ctx);
    REGISTER_API(TensorGetValueAsDouble, ctx);
    REGISTER_API(TensorGetValueAsLongLong, ctx);
    REGISTER_API(TensorGetShallowCopy, ctx);
    REGISTER_API(TensorNumDims, ctx);
    REGISTER_API(TensorDim, ctx);
    REGISTER_API(TensorByteSize, ctx);
    REGISTER_API(TensorData, ctx);
    REGISTER_API(TensorRedisType, ctx);

    REGISTER_API(ModelCreate, ctx);
    REGISTER_API(ModelFree, ctx);
    REGISTER_API(ModelRunCtxCreate, ctx);
    REGISTER_API(GetModelFromKeyspace, ctx);
    REGISTER_API(ModelRunCtxAddInput, ctx);
    REGISTER_API(ModelRunCtxAddOutput, ctx);
    REGISTER_API(ModelRunCtxNumOutputs, ctx);
    REGISTER_API(ModelRunCtxOutputTensor, ctx);
    REGISTER_API(ModelRunCtxFree, ctx);
    REGISTER_API(ModelRun, ctx);
    REGISTER_API(ModelSerialize, ctx);
    REGISTER_API(ModelGetShallowCopy, ctx);
    REGISTER_API(ModelRedisType, ctx);
    REGISTER_API(ModelRunAsync, ctx);
    REGISTER_API(GetAsModelRunCtx, ctx);

    REGISTER_API(ScriptCreate, ctx);
    REGISTER_API(GetScriptFromKeyspace, ctx);
    REGISTER_API(ScriptFree, ctx);
    REGISTER_API(ScriptRunCtxCreate, ctx);
    REGISTER_API(ScriptRunCtxAddInput, ctx);
    REGISTER_API(ScriptRunCtxAddInputList, ctx);
    REGISTER_API(ScriptRunCtxAddOutput, ctx);
    REGISTER_API(ScriptRunCtxNumOutputs, ctx);
    REGISTER_API(ScriptRunCtxOutputTensor, ctx);
    REGISTER_API(ScriptRunCtxFree, ctx);
    REGISTER_API(ScriptRun, ctx);
    REGISTER_API(ScriptGetShallowCopy, ctx);
    REGISTER_API(ScriptRedisType, ctx);
    REGISTER_API(ScriptRunAsync, ctx);
    REGISTER_API(GetAsScriptRunCtx, ctx);

    REGISTER_API(DAGRunCtxCreate, ctx);
    REGISTER_API(DAGCreateModelRunOp, ctx);
    REGISTER_API(DAGCreateScriptRunOp, ctx);
    REGISTER_API(DAGRunOpAddInput, ctx);
    REGISTER_API(DAGRunOpAddOutput, ctx);
    REGISTER_API(DAGAddRunOp, ctx);
    REGISTER_API(DAGLoadTensor, ctx);
    REGISTER_API(DAGAddTensorSet, ctx);
    REGISTER_API(DAGAddTensorGet, ctx);
    REGISTER_API(DAGAddOpsFromString, ctx);
    REGISTER_API(DAGNumOps, ctx);
    REGISTER_API(DAGRun, ctx);
    REGISTER_API(DAGNumOutputs, ctx);
    REGISTER_API(DAGOutputTensor, ctx);
    REGISTER_API(DAGRunError, ctx);
    REGISTER_API(DAGGetError, ctx);
    REGISTER_API(DAGRunOpFree, ctx);
    REGISTER_API(DAGFree, ctx);

    return REDISMODULE_OK;
}

void RAI_moduleInfoFunc(RedisModuleInfoCtx *ctx, int for_crash_report) {
    RedisModule_InfoAddSection(ctx, "git");
    RedisModule_InfoAddFieldCString(ctx, "git_sha", REDISAI_GIT_SHA);
    RedisModule_InfoAddSection(ctx, "load_time_configs");
    RedisModule_InfoAddFieldLongLong(ctx, "threads_per_queue", perqueueThreadPoolSize);
    RedisModule_InfoAddFieldLongLong(ctx, "inter_op_parallelism", getBackendsInterOpParallelism());
    RedisModule_InfoAddFieldLongLong(ctx, "intra_op_parallelism", getBackendsIntraOpParallelism());
    struct rusage self_ru, c_ru;

    // Return resource usage statistics for the calling process,
    // which is the sum of resources used by all threads in the
    // process
    getrusage(RUSAGE_SELF, &self_ru);

    // Return resource usage statistics for the calling thread
    // which in this case is Redis/RedisAI main thread
    // RUSAGE_THREAD is Linux-specific.
    RedisModuleString *main_thread_used_cpu_sys = NULL;
    RedisModuleString *main_thread_used_cpu_user = NULL;
#if (defined(__linux__) && defined(RUSAGE_THREAD))
    struct rusage main_thread_ru;
    getrusage(RUSAGE_THREAD, &main_thread_ru);
    main_thread_used_cpu_sys = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)main_thread_ru.ru_stime.tv_sec, (long)self_ru.ru_stime.tv_usec);
    main_thread_used_cpu_user = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)main_thread_ru.ru_utime.tv_sec, (long)self_ru.ru_utime.tv_usec);
#else
    main_thread_used_cpu_sys = RedisModule_CreateStringPrintf(NULL, "N/A");
    main_thread_used_cpu_user = RedisModule_CreateStringPrintf(NULL, "N/A");
#endif

    // Return resource usage statistics for all of its
    // terminated child processes
    getrusage(RUSAGE_CHILDREN, &c_ru);
    RedisModuleString *self_used_cpu_sys = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)self_ru.ru_stime.tv_sec, (long)self_ru.ru_stime.tv_usec);
    RedisModuleString *self_used_cpu_user = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)self_ru.ru_utime.tv_sec, (long)self_ru.ru_utime.tv_usec);
    RedisModuleString *children_used_cpu_sys = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)c_ru.ru_stime.tv_sec, (long)c_ru.ru_stime.tv_usec);
    RedisModuleString *children_used_cpu_user = RedisModule_CreateStringPrintf(
        NULL, "%ld.%06ld", (long)c_ru.ru_utime.tv_sec, (long)c_ru.ru_utime.tv_usec);
    RedisModule_InfoAddSection(ctx, "cpu");
    RedisModule_InfoAddFieldString(ctx, "self_used_cpu_sys", self_used_cpu_sys);
    RedisModule_InfoAddFieldString(ctx, "self_used_cpu_user", self_used_cpu_user);
    RedisModule_InfoAddFieldString(ctx, "children_used_cpu_sys", children_used_cpu_sys);
    RedisModule_InfoAddFieldString(ctx, "children_used_cpu_user", children_used_cpu_user);
    RedisModule_InfoAddFieldString(ctx, "main_thread_used_cpu_sys", main_thread_used_cpu_sys);
    RedisModule_InfoAddFieldString(ctx, "main_thread_used_cpu_user", main_thread_used_cpu_user);

    RedisModule_FreeString(NULL, self_used_cpu_sys);
    RedisModule_FreeString(NULL, self_used_cpu_user);
    RedisModule_FreeString(NULL, children_used_cpu_sys);
    RedisModule_FreeString(NULL, children_used_cpu_user);
    RedisModule_FreeString(NULL, main_thread_used_cpu_sys);
    RedisModule_FreeString(NULL, main_thread_used_cpu_user);

    AI_dictIterator *iter = AI_dictGetSafeIterator(run_queues);
    AI_dictEntry *entry = AI_dictNext(iter);
    while (entry) {
        char *queue_name = (char *)AI_dictGetKey(entry);
        RunQueueInfo *run_queue_info = (RunQueueInfo *)AI_dictGetVal(entry);
        if (run_queue_info) {
            for (int i = 0; i < perqueueThreadPoolSize; i++) {
                pthread_t current_bg_threads = run_queue_info->threads[i];
                struct timespec ts;
                clockid_t cid;
                RedisModuleString *queue_used_cpu_total = RedisModule_CreateStringPrintf(
                    NULL, "queue_%s_bthread_n%d_used_cpu_total", queue_name, i + 1);
                RedisModuleString *bthread_used_cpu_total = NULL;
#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE) ||         \
    defined(__cplusplus)
                const int status = -1;
#else
                const int status = pthread_getcpuclockid(current_bg_threads, &cid);
#endif
                if (status != 0) {
                    bthread_used_cpu_total = RedisModule_CreateStringPrintf(NULL, "N/A");
                } else {
                    if (clock_gettime(cid, &ts) == -1) {
                        bthread_used_cpu_total = RedisModule_CreateStringPrintf(NULL, "N/A");
                    } else {
                        bthread_used_cpu_total = RedisModule_CreateStringPrintf(
                            NULL, "%ld.%06ld", (long)ts.tv_sec, (long)(ts.tv_nsec / 1000));
                    }
                }
                RedisModule_InfoAddFieldString(
                    ctx, (char *)RedisModule_StringPtrLen(queue_used_cpu_total, NULL),
                    bthread_used_cpu_total);
                RedisModule_FreeString(NULL, queue_used_cpu_total);
                RedisModule_FreeString(NULL, bthread_used_cpu_total);
            }
        }
        entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    if (RedisModule_Init(ctx, "ai", REDISAI_MODULE_VERSION, REDISMODULE_APIVER_1) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;

    getRedisVersion();
    RedisModule_Log(ctx, "notice", "Redis version found by RedisAI: %d.%d.%d - %s",
                    redisMajorVersion, redisMinorVersion, redisPatchVersion,
                    IsEnterprise() ? "enterprise" : "oss");
    if (IsEnterprise()) {
        RedisModule_Log(ctx, "notice", "Redis Enterprise version found by RedisAI: %d.%d.%d-%d",
                        rlecMajorVersion, rlecMinorVersion, rlecPatchVersion, rlecBuild);
    }

    if (redisMajorVersion < 5 ||
        (redisMajorVersion == 5 && redisMinorVersion == 0 && redisPatchVersion < 7)) {
        RedisModule_Log(ctx, "warning",
                        "RedisAI requires Redis version equal or greater than 5.0.7");
        return REDISMODULE_ERR;
    }

    if (redisMajorVersion >= 6) {
        if (RedisModule_RegisterInfoFunc(ctx, RAI_moduleInfoFunc) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
    }

    RedisModule_Log(ctx, "notice", "RedisAI version %d, git_sha=%s", REDISAI_MODULE_VERSION,
                    REDISAI_GIT_SHA);

    int flags = RedisModule_GetContextFlags(ctx);

    if (RedisAI_RegisterApi(ctx) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
        return REDISMODULE_ERR;
    }

    if (!TensorType_Register(ctx)) {
        RedisModule_Log(ctx, "warning", "can not initialize tensor dt\r\n");
        return REDISMODULE_ERR;
    }

    if (!ModelType_Register(ctx)) {
        RedisModule_Log(ctx, "warning", "can not initialize model dt\r\n");
        return REDISMODULE_ERR;
    }

    if (!ScriptType_Register(ctx)) {
        RedisModule_Log(ctx, "warning", "can not initialize script dt\r\n");
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "ai.tensorset", RedisAI_TensorSet_RedisCommand,
                                  "write deny-oom", 1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.tensorget", RedisAI_TensorGet_RedisCommand, "readonly",
                                  1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.modelset", RedisAI_ModelSet_RedisCommand,
                                  "write deny-oom", 1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.modelget", RedisAI_ModelGet_RedisCommand, "readonly", 1,
                                  1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.modeldel", RedisAI_ModelDel_RedisCommand, "write", 1, 1,
                                  1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.modelrun", RedisAI_ModelRun_RedisCommand,
                                  "write deny-oom getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai._modelscan", RedisAI_ModelScan_RedisCommand, "readonly",
                                  1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.scriptset", RedisAI_ScriptSet_RedisCommand,
                                  "write deny-oom", 1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.scriptget", RedisAI_ScriptGet_RedisCommand, "readonly",
                                  1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.scriptdel", RedisAI_ScriptDel_RedisCommand, "write", 1,
                                  1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.scriptrun", RedisAI_ScriptRun_RedisCommand,
                                  "write deny-oom getkeys-api", 4, 4, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai._scriptscan", RedisAI_ScriptScan_RedisCommand,
                                  "readonly", 1, 1, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.info", RedisAI_Info_RedisCommand, "readonly", 1, 1, 1) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.config", RedisAI_Config_RedisCommand, "write", 1, 1,
                                  1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.dagrun", RedisAI_DagRun_RedisCommand,
                                  "write deny-oom getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.dagrun_ro", RedisAI_DagRunRO_RedisCommand,
                                  "readonly getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    RedisModule_SetModuleOptions(ctx, REDISMODULE_OPTIONS_HANDLE_IO_ERRORS);

    // Default configs
    RAI_BackendsPath = NULL;
    perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    setBackendsInterOpParallelism(REDISAI_DEFAULT_INTER_OP_PARALLELISM);
    setBackendsIntraOpParallelism(REDISAI_DEFAULT_INTRA_OP_PARALLELISM);
    setModelChunkSize(REDISAI_DEFAULT_MODEL_CHUNK_SIZE);

    RAI_loadTimeConfig(ctx, argv, argc);

    run_queues = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
    RunQueueInfo *run_queue_info = NULL;
    if (ensureRunQueue("CPU", &run_queue_info) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "Queue not initialized for device CPU");
        return REDISMODULE_ERR;
    }

    run_stats = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);

    return REDISMODULE_OK;
}
