#include "deprecated.h"
#include "rmutil/args.h"
#include "backends/backends.h"
#include "util/string_utils.h"
#include "redis_ai_objects/stats.h"

#include "execution/utils.h"
#include <execution/run_queue_info.h>
#include "execution/DAG/dag_builder.h"
#include "execution/DAG/dag_execute.h"
#include "execution/parsing/dag_parser.h"
#include "execution/parsing/parse_utils.h"
#include "execution/parsing/tensor_commands_parsing.h"

#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/execution_contexts/scriptRun_ctx.h"

static int _ModelRunCommand_ParseArgs(RedisModuleCtx *ctx, int argc, RedisModuleString **argv,
                                      RAI_Model **model, RAI_Error *error,
                                      RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                      long long *timeout) {

    if (argc < 6) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR wrong number of arguments for 'AI.MODELRUN' command");
        return REDISMODULE_ERR;
    }
    size_t argpos = 1;
    const int status = RAI_GetModelFromKeyspace(ctx, argv[argpos], model, REDISMODULE_READ, error);
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    const char *arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);

    // Parse timeout arg if given and store it in timeout
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (ParseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
        arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);
    }
    if (strcasecmp(arg_string, "INPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
        return REDISMODULE_ERR;
    }

    bool is_input = true, is_output = false;
    size_t ninputs = 0, noutputs = 0;

    while (++argpos < argc) {
        arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "OUTPUTS") && !is_output) {
            is_input = false;
            is_output = true;
        } else {
            RAI_HoldString(argv[argpos]);
            if (is_input) {
                ninputs++;
                *inkeys = array_append(*inkeys, argv[argpos]);
            } else {
                noutputs++;
                *outkeys = array_append(*outkeys, argv[argpos]);
            }
        }
    }
    if ((*model)->ninputs != ninputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as INPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }

    if ((*model)->noutputs != noutputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as OUTPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ParseModelRunCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                         int argc) {

    int res = REDISMODULE_ERR;
    // Build a ModelRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RAI_Model *model;
    long long timeout = 0;
    if (_ModelRunCommand_ParseArgs(ctx, argc, argv, &model, rinfo->err, &currentOp->inkeys,
                                   &currentOp->outkeys, &timeout) == REDISMODULE_ERR) {
        goto cleanup;
    }

    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    RAI_ModelRunCtx *mctx = RAI_ModelRunCtxCreate(model);
    currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
    currentOp->ectx = (RAI_ExecutionCtx *)mctx;
    currentOp->devicestr = mctx->model->devicestr;

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ModelRunCtx, bring inputs from key space.
        if (ModelRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, mctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}

int ModelSetCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
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

        if (AC_IsAtEnd(&optionsac) || !AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
            return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified");
        }
        AC_GetSliceToEnd(&optionsac, &outac);
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
        .backends_intra_op_parallelism = Config_GetBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = Config_GetBackendsInterOpParallelism(),
    };

    RAI_Model *model = NULL;

    AC_AdvanceUntilMatches(&ac, 1, blob_matches);

    if (AC_Advance(&ac) != AC_OK || AC_IsAtEnd(&ac)) {
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

    if (!RunQueue_IsExists(devicestr)) {
        RunQueueInfo *run_queue_info = RunQueue_Create(devicestr);
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

    RAI_RunStats *stats = RAI_StatsCreate(keystr, RAI_MODEL, backend, devicestr, tag);
    RAI_StatsStoreEntry(keystr, stats);
    model->info = stats;

    RedisModule_CloseKey(key);
    RedisModule_ReplyWithSimpleString(ctx, "OK");
    RedisModule_ReplicateVerbatim(ctx);

    return REDISMODULE_OK;
}

int ScriptSetCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
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

    if (!RunQueue_IsExists(devicestr)) {
        RunQueueInfo *run_queue_info = RunQueue_Create(devicestr);
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

    RAI_RunStats *stats = RAI_StatsCreate(keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);
    RAI_StatsStoreEntry(keystr, stats);
    script->info = stats;

    RedisModule_CloseKey(key);
    RedisModule_ReplyWithSimpleString(ctx, "OK");
    RedisModule_ReplicateVerbatim(ctx);

    return REDISMODULE_OK;
}

static int _ScriptRunCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                       RAI_Error *error, RedisModuleString ***inkeys,
                                       RedisModuleString ***outkeys, long long *timeout) {

    bool is_input = false;
    bool is_output = false;
    bool timeout_set = false;
    bool inputs_done = false;
    size_t ninputs = 0, noutputs = 0;
    int varidic_start_pos = -1;
    for (int argpos = 3; argpos < argc; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);

        // Parse timeout arg if given and store it in timeout
        if (!strcasecmp(arg_string, "TIMEOUT")) {
            if (timeout_set) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an TIMEOUT section in SCRIPTRUN");
                return REDISMODULE_ERR;
            }
            if (ParseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
                return REDISMODULE_ERR;
            timeout_set = true;
            continue;
        }

        if (!strcasecmp(arg_string, "INPUTS")) {
            if (inputs_done) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an INPUTS section in SCRIPTRUN");
                return REDISMODULE_ERR;
            }
            if (is_input) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an INPUTS keyword in SCRIPTRUN");
                return REDISMODULE_ERR;
            }
            is_input = true;
            is_output = false;
            continue;
        }
        if (!strcasecmp(arg_string, "OUTPUTS")) {
            if (is_output) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an OUTPUTS keyword in SCRIPTRUN");
                return REDISMODULE_ERR;
            }
            is_input = false;
            is_output = true;
            inputs_done = true;
            continue;
        }
        if (!strcasecmp(arg_string, "$")) {
            if (!is_input) {
                RAI_SetError(
                    error, RAI_ESCRIPTRUN,
                    "ERR Encountered a variable size list of tensors outside of input section");
                return REDISMODULE_ERR;
            }
            if (varidic_start_pos > -1) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered a variable size list of tensors");
                return REDISMODULE_ERR;
            }
            varidic_start_pos = ninputs;
            continue;
        }
        // Parse argument name
        if (is_input) {
            ninputs++;
            *inkeys = array_append(*inkeys, RAI_HoldString(argv[argpos]));
        } else if (is_output) {
            noutputs++;
            *outkeys = array_append(*outkeys, RAI_HoldString(argv[argpos]));
        } else {
            RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Unrecongnized parameter to SCRIPTRUN");
            return REDISMODULE_ERR;
        }
    }

    return REDISMODULE_OK;
}

int ParseScriptRunCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                          int argc) {

    if (argc < 3) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN,
                     "ERR wrong number of arguments for 'AI.SCRIPTRUN' command");
        return REDISMODULE_ERR;
    }

    int res = REDISMODULE_ERR;
    // Build a ScriptRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RAI_ScriptRunCtx *sctx = NULL;
    RAI_Script *script = NULL;
    RedisModuleString *scriptName = argv[1];
    RAI_GetScriptFromKeyspace(ctx, scriptName, &script, REDISMODULE_READ, rinfo->err);
    if (!script) {
        goto cleanup;
    }

    const char *func_name = ScriptCommand_GetFunctionName(argv[2]);
    if (!func_name) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN, "ERR function name not specified");
        goto cleanup;
    }

    sctx = RAI_ScriptRunCtxCreate(script, func_name);
    long long timeout = 0;
    if (_ScriptRunCommand_ParseArgs(ctx, argv, argc, rinfo->err, &currentOp->inkeys,
                                    &currentOp->outkeys, &timeout) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ScriptRunCtx, bring inputs from key space.
        if (ScriptRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, sctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    currentOp->ectx = (RAI_ExecutionCtx *)sctx;
    currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    currentOp->devicestr = sctx->script->devicestr;
    res = REDISMODULE_OK;
    RedisModule_FreeThreadSafeContext(ctx);
    return res;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    if (sctx) {
        RAI_ScriptRunCtxFree(sctx);
    }
    return res;
}

int ParseDAGRunOps(RedisAI_RunInfo *rinfo, RAI_DagOp **ops) {

    for (long long i = 0; i < array_len(ops); i++) {
        RAI_DagOp *currentOp = ops[i];
        // The first op arg is the command name.
        const char *arg_string = RedisModule_StringPtrLen(currentOp->argv[0], NULL);

        if (!strcasecmp(arg_string, "AI.TENSORGET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORGET;
            currentOp->devicestr = "CPU";
            currentOp->fmt = ParseTensorGetFormat(rinfo->err, currentOp->argv, currentOp->argc);
            if (currentOp->fmt == TENSOR_NONE)
                goto cleanup;
            RAI_HoldString(currentOp->argv[1]);
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            continue;
        }
        if (!strcasecmp(arg_string, "AI.TENSORSET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORSET;
            currentOp->devicestr = "CPU";
            if (ParseTensorSetArgs(currentOp->argv, currentOp->argc, &currentOp->outTensor,
                                   rinfo->err) != REDISMODULE_OK) {
                goto cleanup;
            }
            RAI_HoldString(currentOp->argv[1]);
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            currentOp->result = REDISMODULE_OK;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELRUN")) {
            if (ParseModelRunCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                goto cleanup;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
            if (ParseScriptRunCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                goto cleanup;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELEXECUTE")) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "AI.MODELEXECUTE"
                         " cannot be used in a deprecated AI.DAGRUN command");
            goto cleanup;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTEXECUTE")) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "AI.SCRIPTEXECUTE"
                         " cannot be used in a deprecated AI.DAGRUN command");
            goto cleanup;
        }
        // If none of the cases match, we have an invalid op.
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "Unsupported command within DAG");
        goto cleanup;
    }

    // After validating all the ops, insert them to the DAG.
    for (size_t i = 0; i < array_len(ops); i++) {
        rinfo->dagOps = array_append(rinfo->dagOps, ops[i]);
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    return REDISMODULE_OK;

cleanup:
    for (size_t i = 0; i < array_len(ops); i++) {
        RAI_FreeDagOp(ops[i]);
    }
    return REDISMODULE_ERR;
}

int ParseDAGRunCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                       int argc, bool dag_ro) {

    int res = REDISMODULE_ERR;
    // This minimal possible command (syntactically) is: AI.DAGRUN(_RO) |> TENSORGET <key>.
    if (argc < 4) {
        if (dag_ro) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR wrong number of arguments for 'AI.DAGRUN_RO' command");
        } else {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR wrong number of arguments for 'AI.DAGRUN' command");
        }
        return res;
    }

    // First we parse LOAD, PERSIST and TIMEOUT parts, and we collect the DAG ops' args.
    array_new_on_stack(RAI_DagOp *, 10, dag_ops);
    if (DAGInitialParsing(rinfo, ctx, argv, argc, dag_ro, &dag_ops) != REDISMODULE_OK) {
        goto cleanup;
    }

    if (ParseDAGRunOps(rinfo, dag_ops) != REDISMODULE_OK) {
        goto cleanup;
    }
    if (MapTensorsKeysToIndices(rinfo, rinfo->tensorsNamesToIndices) != REDISMODULE_OK) {
        goto cleanup;
    }
    if (ValidatePersistKeys(rinfo, rinfo->tensorsNamesToIndices, rinfo->persistTensors) !=
        REDISMODULE_OK) {
        goto cleanup;
    }

    AI_dictRelease(rinfo->tensorsNamesToIndices);
    rinfo->tensorsNamesToIndices = NULL;
    res = REDISMODULE_OK;

cleanup:
    array_free(dag_ops);
    return res;
}
