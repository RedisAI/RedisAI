
#ifdef __linux__
#define _GNU_SOURCE
#endif

#define REDISMODULE_MAIN
#include "redismodule.h"
#include "redis_ai_objects/tensor.h"
#include "execution/command_parser.h"
#include "backends/backends.h"
#include "backends/util.h"
#include "execution/utils.h"
#include "execution/background_workers.h"
#include "execution/DAG/dag.h"
#include "execution/DAG/dag_builder.h"
#include "execution/DAG/dag_execute.h"
#include "execution/parsing/deprecated.h"
#include "redis_ai_objects/model.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/execution_contexts/scriptRun_ctx.h"
#include "redis_ai_objects/script.h"
#include "redis_ai_objects/stats.h"
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <execution/onnx_timeout.h>

#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "util/arr.h"
#include "util/dictionaries.h"
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

extern int redisMajorVersion;
extern int redisMinorVersion;
extern int redisPatchVersion;

extern int rlecMajorVersion;
extern int rlecMinorVersion;
extern int rlecPatchVersion;
extern int rlecBuild;

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

    if (RedisModule_ModuleTypeSetValue(key, RAI_TensorRedisType(), t) != REDISMODULE_OK) {
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

    RedisModule_Log(ctx, "warning",
                    "AI.MODELSET command is deprecated and will"
                    " not be available in future version, you can use AI.MODELSTORE instead");
    return ModelSetCommand(ctx, argv, argc);
}

/**
 * AI.MODELSTORE model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]]
 * [INPUTS input_count name1 name2 ... OUTPUTS output_count name1 name2 ...] BLOB model_blob
 */
int RedisAI_ModelStore_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 6)
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

    // Parse <backend> argument: check that the device string is "CPU", "GPU",
    // "CPU:<n>" or "GPU:<n>, where <n> is a number (contains digits only).
    const char *devicestr;
    AC_GetString(&ac, &devicestr, NULL, 0);
    bool valid_device = false;
    if (strcasecmp(devicestr, "CPU") == 0 || strcasecmp(devicestr, "GPU") == 0) {
        valid_device = true;
    } else if ((strncasecmp(devicestr, "GPU:", 4) == 0 || strncasecmp(devicestr, "CPU:", 4) == 0) &&
               strlen(devicestr) <= 10) {
        bool digits_only = true;
        for (size_t i = 5; i < strlen(devicestr); i++) {
            if (devicestr[i] < '0' || devicestr[i] > '9') {
                digits_only = false;
                break;
            }
        }
        valid_device = digits_only;
    }
    if (!valid_device) {
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
        if (minbatchsize == 0) {
            return RedisModule_ReplyWithError(ctx,
                                              "ERR MINBATCHTIMEOUT specified without MINBATCHSIZE");
        }
        if (AC_GetUnsignedLongLong(&ac, &minbatchtimeout, 0) != AC_OK) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHTIMEOUT");
        }
    }
    RAI_ModelOpts opts = {
        .batchsize = batchsize,
        .minbatchsize = minbatchsize,
        .minbatchtimeout = minbatchtimeout,
        .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
    };

    if (AC_IsAtEnd(&ac)) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
    }
    const char *arg_string;
    AC_GetString(&ac, &arg_string, NULL, 0);
    unsigned long long ninputs = 0, noutputs = 0;

    if (backend == RAI_BACKEND_TENSORFLOW) {
        if (strcasecmp(arg_string, "INPUTS") != 0) {
            return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified for TF");
        }
        if (AC_GetUnsignedLongLong(&ac, &ninputs, 0) != AC_OK || ninputs == 0) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for input_count");
        }
        if (AC_NumRemaining(&ac) < ninputs) {
            return RedisModule_ReplyWithError(
                ctx, "ERR number of model inputs does not match the number of "
                     "given arguments");
        }
    } else if (strcasecmp(arg_string, "INPUTS") == 0) {
        return RedisModule_ReplyWithError(
            ctx, "ERR INPUTS argument should not be specified for this backend");
    }
    const char *inputs[ninputs];
    for (size_t i = 0; i < ninputs; i++) {
        AC_GetString(&ac, inputs + i, NULL, 0);
    }

    if (backend == RAI_BACKEND_TENSORFLOW) {
        if (AC_GetString(&ac, &arg_string, NULL, 0) != AC_OK ||
            strcasecmp(arg_string, "OUTPUTS") != 0) {
            return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified for TF");
        }
        if (AC_GetUnsignedLongLong(&ac, &noutputs, 0) != AC_OK || noutputs == 0) {
            return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for output_count");
        }
        if (AC_NumRemaining(&ac) < noutputs) {
            return RedisModule_ReplyWithError(
                ctx, "ERR number of model outputs does not match the number of "
                     "given arguments");
        }
    }
    const char *outputs[noutputs];
    for (size_t i = 0; i < noutputs; i++) {
        AC_GetString(&ac, outputs + i, NULL, 0);
    }
    if (backend == RAI_BACKEND_TENSORFLOW) {
        AC_GetString(&ac, &arg_string, NULL, 0);
    }

    if (strcasecmp(arg_string, "BLOB") != 0) {
        return RedisModule_ReplyWithError(ctx, "ERR Invalid argument, expected BLOB");
    }

    if (AC_IsAtEnd(&ac)) {
        return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
    }

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
    RAI_Model *model = NULL;
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                            modeldef, modellen, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend",
                        bckstr);
        int ret = RAI_LoadDefaultBackend(ctx, backend);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "could not load %s default backend", bckstr);
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
        RedisModule_Log(ctx, "warning", "%s", err.detail);
        int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
        RAI_ClearError(&err);
        return ret;
    }

    // TODO: if backend loaded, make sure there's a queue
    if (!IsRunQueueExists(devicestr)) {
        RunQueueInfo *run_queue_info = CreateRunQueue(devicestr);
        if (run_queue_info == NULL) {
            RAI_ModelFree(model, &err);
            RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
        }
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ | REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(key) == RAI_ModelRedisType())) {
        RedisModule_CloseKey(key);
        RAI_ModelFree(model, &err);
        if (err.code != RAI_OK) {
            RedisModule_Log(ctx, "warning", "%s", err.detail);
            int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
            RAI_ClearError(&err);
            return ret;
        }
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }

    RedisModule_ModuleTypeSetValue(key, RAI_ModelRedisType(), model);

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

    const int outentries = blob ? 18 : 16;
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

    RedisModule_ReplyWithCString(ctx, "minbatchtimeout");
    RedisModule_ReplyWithLongLong(ctx, (long)mto->opts.minbatchtimeout);

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
    if (IsEnterprise()) {
        RedisModule_Log(ctx, "warning",
                        "AI.MODELRUN command is deprecated and cannot be used in "
                        "enterprise cluster, use AI.MODELEXECUTE instead");
        return RedisModule_ReplyWithError(
            ctx, "ERR AI.MODELRUN command is deprecated and cannot be used in "
                 "enterprise cluster, use AI.MODELEXECUTE instead");
    } else {
        RedisModule_Log(ctx, "warning",
                        "AI.MODELRUN command is deprecated and will"
                        " not be available in future version, you can use AI.MODELEXECUTE instead");
    }
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_MODELRUN, false);
}

/**
 * AI.MODELEXECUTE <key>
 * INPUTS <input_count> <input> [input ...]
 * OUTPUTS <output_count> <output> [output ...]
 * [TIMEOUT <time>]
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_ModelExecute_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    // This is called only in cluster mode (not enterprise), to ensure that all
    // keys are located at the same shard.
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return ModelExecute_ReportKeysPositions(ctx, argv, argc);
    }

    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_MODELEXECUTE, false);
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
 * AI.SCRIPTEXECUTE <key> <function>
 * KEYS <keys_count> <key> [key ...]
 * [INPUTS <input_count> <input> [input ...] | INPUTS_LIST <list_len> <input> [input ...]]*
 * [OUTPUTS <output_count> <output> [output ...]]
 * [TIMEOUT <time>]
 */
int RedisAI_ScriptExecute_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_ScriptExecute_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }

    // Convert The script run command into a DAG command that contains a single op.
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_SCRIPTEXECUTE, false);
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
            RedisModule_Log(ctx, "warning", "Could not load TORCH default backend");
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

    if (!IsRunQueueExists(devicestr)) {
        RunQueueInfo *run_queue_info = CreateRunQueue(devicestr);
        if (run_queue_info == NULL) {
            RAI_ScriptFree(script, &err);
            RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
        }
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ | REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(key) == RAI_ScriptRedisType())) {
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }

    RedisModule_ModuleTypeSetValue(key, RAI_ScriptRedisType(), script);

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

void _RedisAI_Info(RedisModuleCtx *ctx) {
    RedisModuleString *rai_version = RedisModule_CreateStringPrintf(
        ctx, "%d.%d.%d", REDISAI_VERSION_MAJOR, REDISAI_VERSION_MINOR, REDISAI_VERSION_PATCH);
    RedisModuleString *llapi_version =
        RedisModule_CreateStringPrintf(ctx, "%d", REDISAI_LLAPI_VERSION);
    RedisModuleString *rdb_version = RedisModule_CreateStringPrintf(ctx, "%llu", REDISAI_ENC_VER);

    int reponse_len = 6;

    if (RAI_backends.tf.get_version) {
        reponse_len += 2;
    }

    if (RAI_backends.torch.get_version) {
        reponse_len += 2;
    }

    if (RAI_backends.tflite.get_version) {
        reponse_len += 2;
    }

    if (RAI_backends.onnx.get_version) {
        reponse_len += 2;
    }

    RedisModule_ReplyWithArray(ctx, reponse_len);

    RedisModule_ReplyWithSimpleString(ctx, "Version");
    RedisModule_ReplyWithString(ctx, rai_version);

    // TODO: Add Git SHA

    RedisModule_ReplyWithSimpleString(ctx, "Low Level API Version");
    RedisModule_ReplyWithString(ctx, llapi_version);

    RedisModule_ReplyWithSimpleString(ctx, "RDB Encoding version");
    RedisModule_ReplyWithString(ctx, llapi_version);

    if (RAI_backends.tf.get_version) {
        RedisModule_ReplyWithSimpleString(ctx, "TensorFlow version");
        RedisModule_ReplyWithSimpleString(ctx, RAI_backends.tf.get_version());
    }

    if (RAI_backends.torch.get_version) {
        RedisModule_ReplyWithSimpleString(ctx, "Torch version");
        RedisModule_ReplyWithSimpleString(ctx, RAI_backends.torch.get_version());
    }

    if (RAI_backends.tflite.get_version) {
        RedisModule_ReplyWithSimpleString(ctx, "TFLite version");
        RedisModule_ReplyWithSimpleString(ctx, RAI_backends.tflite.get_version());
    }

    if (RAI_backends.onnx.get_version) {
        RedisModule_ReplyWithSimpleString(ctx, "ONNX version");
        RedisModule_ReplyWithSimpleString(ctx, RAI_backends.onnx.get_version());
    }

    RedisModule_FreeString(ctx, rai_version);
    RedisModule_FreeString(ctx, llapi_version);
    RedisModule_FreeString(ctx, rdb_version);
}

/**
 * AI.INFO <model_or_script_key> [RESETSTAT]
 */
int RedisAI_Info_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc > 3)
        return RedisModule_WrongArity(ctx);

    if (argc == 1) {
        _RedisAI_Info(ctx);
        return REDISMODULE_OK;
    }

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
    if (IsEnterprise()) {
        RedisModule_Log(ctx, "warning",
                        "AI.DAGRUN command is deprecated and cannot be used in "
                        "enterprise cluster, use AI.DAGEXECUTE instead");
        return RedisModule_ReplyWithError(
            ctx, "ERR AI.DAGRUN command is deprecated and cannot be used in "
                 "enterprise cluster, use AI.DAGEXECUTE instead");
    } else {
        RedisModule_Log(ctx, "warning",
                        "AI.DAGRUN command is deprecated and will"
                        " not be available in future version, you can use AI.DAGEXECUTE instead");
    }
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagExecute_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGRUN, false);
}

/**
 * AI.DAGEXECUTE
 * [[KEYS <keys_count> <key> [key ...]] |
 * [LOAD <nkeys> <key> [key... ]] |
 * [PERSIST <nkeys> <key> [key2...]]]+
 * [TIMEOUT t]
 * [|> <COMMAND1>]+
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagExecute_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagExecute_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGEXECUTE, false);
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
    if (IsEnterprise()) {
        RedisModule_Log(ctx, "warning",
                        "AI.DAGRUN_RO command is deprecated and cannot be used in "
                        "enterprise cluster, use AI.DAGEXECUTE_RO instead");
        return RedisModule_ReplyWithError(
            ctx, "ERR AI.DAGRUN_RO command is deprecated and cannot be used in "
                 "enterprise cluster, use AI.DAGEXECUTE instead");
    } else {
        RedisModule_Log(ctx, "warning",
                        "AI.DAGRUN_RO command is deprecated and will"
                        " not be available in future version, you can use AI.DAGEXECUTE instead");
    }
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagExecute_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGRUN, true);
}

/**
 * AI.DAGEXECUTE_RO
 * [[KEYS <keys_count> <key> [key ...]] |
 * [LOAD <nkeys> key [key ... ]]]+
 * [TIMEOUT t]
 * [|> <COMMAND>]+
 *
 * Read-only (no PERSIST) DAG execution. AI.SCRIPTEXEXUTE is not allowed
 * within RO context.
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagExecuteRO_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_IsKeysPositionRequest(ctx)) {
        return RedisAI_DagExecute_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
    }
    return RedisAI_ExecuteCommand(ctx, argv, argc, CMD_DAGEXECUTE, true);
}

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
    REGISTER_API(ScriptRunCtxAddTensorInput, ctx);
    REGISTER_API(ScriptRunCtxAddIntInput, ctx);
    REGISTER_API(ScriptRunCtxAddFloatInput, ctx);
    REGISTER_API(ScriptRunCtxAddRStringInput, ctx);
    REGISTER_API(ScriptRunCtxAddStringInput, ctx);
    REGISTER_API(ScriptRunCtxAddInputList, ctx);
    REGISTER_API(ScriptRunCtxAddTensorInputList, ctx);
    REGISTER_API(ScriptRunCtxAddIntInputList, ctx);
    REGISTER_API(ScriptRunCtxAddFloatInputList, ctx);
    REGISTER_API(ScriptRunCtxAddRStringInputList, ctx);
    REGISTER_API(ScriptRunCtxAddStringInputList, ctx);
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
    RedisModule_InfoAddFieldLongLong(ctx, "threads_per_queue", ThreadPoolSizePerQueue);
    RedisModule_InfoAddFieldLongLong(ctx, "inter_op_parallelism", getBackendsInterOpParallelism());
    RedisModule_InfoAddFieldLongLong(ctx, "intra_op_parallelism", getBackendsIntraOpParallelism());
    RedisModule_InfoAddSection(ctx, "memory_usage");
    if (RAI_backends.onnx.get_memory_info) {
        RedisModule_InfoAddFieldULongLong(ctx, "onnxruntime_memory",
                                          RAI_backends.onnx.get_memory_info());
        RedisModule_InfoAddFieldULongLong(ctx, "onnxruntime_memory_access_num",
                                          RAI_backends.onnx.get_memory_access_num());
    } else {
        RedisModule_InfoAddFieldULongLong(ctx, "onnxruntime_memory", 0);
        RedisModule_InfoAddFieldULongLong(ctx, "onnxruntime_memory_access_num", 0);
    }

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

    AI_dictIterator *iter = AI_dictGetSafeIterator(RunQueues);
    AI_dictEntry *entry = AI_dictNext(iter);
    while (entry) {
        char *queue_name = (char *)AI_dictGetKey(entry);
        RunQueueInfo *run_queue_info = (RunQueueInfo *)AI_dictGetVal(entry);
        if (run_queue_info) {
            for (int i = 0; i < ThreadPoolSizePerQueue; i++) {
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

#ifndef REDISAI_LITE
    if (RedisModule_Init(ctx, "ai", REDISAI_MODULE_VERSION, REDISMODULE_APIVER_1) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;
#else
    if (RedisModule_Init(ctx, "ai-lite", REDISAI_MODULE_VERSION, REDISMODULE_APIVER_1) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;
#endif

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

    if (RedisModule_CreateCommand(ctx, "ai.modelstore", RedisAI_ModelStore_RedisCommand,
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

    if (RedisModule_CreateCommand(ctx, "ai.modelexecute", RedisAI_ModelExecute_RedisCommand,
                                  "write deny-oom getkeys-api", 4, 4, 1) == REDISMODULE_ERR)
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

    if (RedisModule_CreateCommand(ctx, "ai.scriptexecute", RedisAI_ScriptExecute_RedisCommand,
                                  "write deny-oom getkeys-api", 5, 5, 1) == REDISMODULE_ERR)
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

    if (RedisModule_CreateCommand(ctx, "ai.dagexecute", RedisAI_DagExecute_RedisCommand,
                                  "write deny-oom getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.dagrun_ro", RedisAI_DagRunRO_RedisCommand,
                                  "readonly getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "ai.dagexecute_ro", RedisAI_DagExecuteRO_RedisCommand,
                                  "readonly getkeys-api", 3, 3, 1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    RedisModule_SetModuleOptions(ctx, REDISMODULE_OPTIONS_HANDLE_IO_ERRORS);

    // Default configs
    RAI_BackendsPath = NULL;
    ThreadPoolSizePerQueue = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    setBackendsInterOpParallelism(REDISAI_DEFAULT_INTER_OP_PARALLELISM);
    setBackendsIntraOpParallelism(REDISAI_DEFAULT_INTRA_OP_PARALLELISM);
    setModelChunkSize(REDISAI_DEFAULT_MODEL_CHUNK_SIZE);

    RAI_loadTimeConfig(ctx, argv, argc);

    RunQueues = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
    pthread_key_create(&thread_id_key, RedisModule_Free);
    RunQueueInfo *cpu_run_queue_info = CreateRunQueue("CPU");
    if (cpu_run_queue_info == NULL) {
        RedisModule_Log(ctx, "warning", "RedisAI could not initialize run queue for CPU");
        return REDISMODULE_ERR;
    }
    run_stats = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);

    return REDISMODULE_OK;
}
