#include "redismodule.h"
#include "run_info.h"
#include "command_parser.h"
#include "DAG/dag.h"
#include "DAG/dag_parser.h"
#include "util/string_utils.h"
#include "execution/modelRun_ctx.h"
#include "rmutil/args.h"
#include "redis_ai_objects/stats.h"

static int _parseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout) {

    const int retval = RedisModule_StringToLongLong(timeout_arg, timeout);
    if (retval != REDISMODULE_OK || timeout <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static int _ModelRunCommand_ParseArgs(RedisModuleCtx *ctx, int argc, RedisModuleString **argv,
                                      RAI_Model **model, RAI_Error *error,
                                      RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                      RedisModuleString **runkey, long long *timeout) {

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
    RAI_HoldString(NULL, argv[argpos]);
    *runkey = argv[argpos];
    const char *arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);

    // Parse timeout arg if given and store it in timeout
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (_parseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
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
            RAI_HoldString(NULL, argv[argpos]);
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

/**
 * Extract the params for the ModelCtxRun object from AI.MODELRUN arguments.
 *
 * @param ctx Context in which Redis modules operate
 * @param inkeys Model input tensors keys, as an array of strings
 * @param outkeys Model output tensors keys, as an array of strings
 * @param mctx Destination Model context to store the parsed data
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise
 */

static int _ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                                  RedisModuleString **outkeys, RAI_ModelRunCtx *mctx,
                                  RAI_Error *err) {

    RAI_Model *model = mctx->model;
    RAI_Tensor *t;
    RedisModuleKey *key;
    char *opname = NULL;
    size_t ninputs = array_len(inkeys), noutputs = array_len(outkeys);
    for (size_t i = 0; i < ninputs; i++) {
        const int status =
            RAI_GetTensorFromKeyspace(ctx, inkeys[i], &key, &t, REDISMODULE_READ, err);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "could not load input tensor %s from keyspace",
                            RedisModule_StringPtrLen(inkeys[i], NULL));
            return REDISMODULE_ERR;
        }
        if (model->inputs)
            opname = model->inputs[i];
        RAI_ModelRunCtxAddInput(mctx, opname, t);
    }

    for (size_t i = 0; i < noutputs; i++) {
        if (model->outputs)
            opname = model->outputs[i];
        RAI_ModelRunCtxAddOutput(mctx, opname);
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
                                   &currentOp->outkeys, &currentOp->runkey,
                                   &timeout) == REDISMODULE_ERR) {
        goto cleanup;
    }

    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    RAI_ModelRunCtx *mctx = RAI_ModelRunCtxCreate(model);
    currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
    currentOp->mctx = mctx;
    currentOp->devicestr = mctx->model->devicestr;

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ModelRunCtx, bring inputs from key space.
        if (_ModelRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, mctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}

static int _ScriptRunCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                       RAI_Script **script, RAI_Error *error,
                                       RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                       RedisModuleString **runkey, char const **func_name,
                                       long long *timeout, int *variadic) {

    if (argc < 3) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR wrong number of arguments for 'AI.SCRIPTRUN' command");
        return REDISMODULE_ERR;
    }
    size_t argpos = 1;
    const int status =
        RAI_GetScriptFromKeyspace(ctx, argv[argpos], script, REDISMODULE_READ, error);
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    RAI_HoldString(NULL, argv[argpos]);
    *runkey = argv[argpos];

    const char *arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);
    if (!strcasecmp(arg_string, "TIMEOUT") || !strcasecmp(arg_string, "INPUTS") ||
        !strcasecmp(arg_string, "OUTPUTS")) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR function name not specified");
        return REDISMODULE_ERR;
    }
    *func_name = arg_string;

    bool is_input = false;
    bool is_output = false;
    bool timeout_set = false;
    bool inputs_done = false;
    size_t ninputs = 0, noutputs = 0;
    int varidic_start_pos = -1;

    while (++argpos < argc) {
        arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);

        // Parse timeout arg if given and store it in timeout
        if (!strcasecmp(arg_string, "TIMEOUT") && !timeout_set) {
            if (_parseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
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
        RAI_HoldString(NULL, argv[argpos]);
        if (is_input) {
            ninputs++;
            *inkeys = array_append(*inkeys, argv[argpos]);
        } else if (is_output) {
            noutputs++;
            *outkeys = array_append(*outkeys, argv[argpos]);
        } else {
            RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Unrecongnized parameter to SCRIPTRUN");
            return REDISMODULE_ERR;
        }
    }
    *variadic = varidic_start_pos;

    return REDISMODULE_OK;
}

/**
 * Extract the params for the ScriptCtxRun object from AI.SCRIPTRUN arguments.
 *
 * @param ctx Context in which Redis modules operate.
 * @param inkeys Script input tensors keys, as an array of strings.
 * @param outkeys Script output tensors keys, as an array of strings.
 * @param sctx Destination Script context to store the parsed data.
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise.
 */

static int _ScriptRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                                   RedisModuleString **outkeys, RAI_ScriptRunCtx *sctx,
                                   RAI_Error *err) {

    RAI_Tensor *t;
    RedisModuleKey *key;
    size_t ninputs = array_len(inkeys), noutputs = array_len(outkeys);
    for (size_t i = 0; i < ninputs; i++) {
        const int status =
            RAI_GetTensorFromKeyspace(ctx, inkeys[i], &key, &t, REDISMODULE_READ, err);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "could not load input tensor %s from keyspace",
                            RedisModule_StringPtrLen(inkeys[i], NULL));
            return REDISMODULE_ERR;
        }
        RAI_ScriptRunCtxAddInput(sctx, t, err);
    }
    for (size_t i = 0; i < noutputs; i++) {
        RAI_ScriptRunCtxAddOutput(sctx);
    }
    return REDISMODULE_OK;
}

int ParseScriptRunCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                          int argc) {

    int res = REDISMODULE_ERR;
    // Build a ScriptRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    // int lock_status = RedisModule_ThreadSafeContextTryLock(ctx);
    RAI_Script *script;
    const char *func_name = NULL;

    long long timeout = 0;
    int variadic = -1;
    if (_ScriptRunCommand_ParseArgs(ctx, argv, argc, &script, rinfo->err, &currentOp->inkeys,
                                    &currentOp->outkeys, &currentOp->runkey, &func_name, &timeout,
                                    &variadic) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(script, func_name);
    sctx->variadic = variadic;
    currentOp->sctx = sctx;
    currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    currentOp->devicestr = sctx->script->devicestr;

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ScriptRunCtx, bring inputs from key space.
        if (_ScriptRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, sctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    // if (lock_status == REDISMODULE_OK) {
    // RedisModule_ThreadSafeContextUnlock(ctx);
    //}
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}

int _ModelSetCommand_ParseBatchingArgs(ArgsCursor *ac, RAI_ModelOpts *opts, int backend,
                                       RAI_Error *err) {
    unsigned long long batchsize = 0;
    if (AC_AdvanceIfMatch(ac, "BATCHSIZE")) {
        if (backend == RAI_BACKEND_TFLITE) {
            RAI_SetError(err, RAI_EMODELCREATE,
                         "ERR Auto-batching not supported by the TFLITE backend");
            return REDISMODULE_ERR;
        }
        if (AC_GetUnsignedLongLong(ac, &batchsize, 0) != AC_OK) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR Invalid argument for BATCHSIZE");
            return REDISMODULE_ERR;
        }
    }

    unsigned long long minbatchsize = 0;
    if (AC_AdvanceIfMatch(ac, "MINBATCHSIZE")) {
        if (batchsize == 0) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR MINBATCHSIZE specified without BATCHSIZE");
            return REDISMODULE_ERR;
        }
        if (AC_GetUnsignedLongLong(ac, &minbatchsize, 0) != AC_OK) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR Invalid argument for MINBATCHSIZE");
            return REDISMODULE_ERR;
        }
    }

    unsigned long long minbatchtimeout = 0;
    if (AC_AdvanceIfMatch(ac, "MINBATCHTIMEOUT")) {
        if (batchsize == 0) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR MINBATCHTIMEOUT specified without BATCHSIZE");
            return REDISMODULE_ERR;
        }
        if (minbatchsize == 0) {
            RAI_SetError(err, RAI_EMODELCREATE,
                         "ERR MINBATCHTIMEOUT specified without MINBATCHSIZE");
            return REDISMODULE_ERR;
        }
        if (AC_GetUnsignedLongLong(ac, &minbatchtimeout, 0) != AC_OK) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR Invalid argument for MINBATCHTIMEOUT");
            return REDISMODULE_ERR;
        }
    }

    opts->batchsize = batchsize;
    opts->minbatchsize = minbatchsize;
    opts->minbatchtimeout = minbatchtimeout;

    return REDISMODULE_OK;
}

int _ModelSetCommand_ParseIOArgs(RAI_Model *model, ArgsCursor *ac, int backend, RAI_Error *err) {

    ArgsCursor optionsac;
    const char *blob_matches[1] = {"BLOB"};
    AC_GetSliceUntilMatches(ac, &optionsac, 1, blob_matches);

    if (optionsac.argc == 0) {
        RAI_SetError(err, RAI_EMODELCREATE,
                     "ERR Insufficient arguments, INPUTS and OUTPUTS not specified for TF model");
        return REDISMODULE_ERR;
    }
    ArgsCursor inac = {0};
    ArgsCursor outac = {0};
    if (optionsac.argc > 0 && backend == RAI_BACKEND_TENSORFLOW) {
        if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
            RAI_SetError(err, RAI_EMODELCREATE, "ERR INPUTS not specified for TF model");
            return REDISMODULE_ERR;
        }

        const char *matches[1] = {"OUTPUTS"};
        AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);
        if (!AC_IsAtEnd(&optionsac)) {
            if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
                RAI_SetError(err, RAI_EMODELCREATE, "ERR OUTPUTS not specified for TF model");
                return REDISMODULE_ERR;
            }
            AC_GetSliceToEnd(&optionsac, &outac);
        }
    }

    model->ninputs = inac.argc;
    model->inputs = array_new(char *, model->ninputs);
    for (size_t i = 0; i < model->ninputs; i++) {
        const char *input_str;
        AC_GetString(&inac, &input_str, NULL, 0);
        model->inputs = array_append(model->inputs, RedisModule_Strdup(input_str));
    }
    model->noutputs = outac.argc;
    model->outputs = array_new(char *, model->noutputs);
    for (size_t i = 0; i < model->noutputs; i++) {
        const char *output_str;
        AC_GetString(&outac, &output_str, NULL, 0);
        model->outputs = array_append(model->outputs, RedisModule_Strdup(output_str));
    }
    return REDISMODULE_OK;
}

void _ModelSetCommand_ParseBlob(RAI_Model *model, ArgsCursor *ac) {
    ArgsCursor blobsac;
    AC_GetSliceToEnd(ac, &blobsac);
    size_t model_len;
    char *model_def;
    char *model_data;

    if (blobsac.argc == 1) {
        AC_GetString(&blobsac, (const char **)&model_def, &model_len, 0);
        model_data = RedisModule_Alloc(model_len);
        memcpy(model_data, model_def, model_len);
    } else {
        // Blobs of large models are chunked, in this case we go over and copy the chunks.
        const char *chunks[blobsac.argc];
        size_t chunk_lens[blobsac.argc];
        model_len = 0;
        while (!AC_IsAtEnd(&blobsac)) {
            AC_GetString(&blobsac, &chunks[blobsac.offset], &chunk_lens[blobsac.offset], 0);
            model_len += chunk_lens[blobsac.offset - 1];
        }
        model_data = RedisModule_Alloc(model_len);
        size_t offset = 0;
        for (size_t i = 0; i < blobsac.argc; i++) {
            memcpy(model_data + offset, chunks[i], chunk_lens[i]);
            offset += chunk_lens[i];
        }
    }
    model->data = model_data;
    model->datalen = model_len;
}

int ParseModelSetCommand(RedisModuleString **argv, int argc, RAI_Model *model, RAI_Error *err) {

    // Use an args cursor object to go over and parse the command args.
    ArgsCursor ac;
    ArgsCursor_InitRString(&ac, argv, argc);

    // Parse model key.
    RedisModuleString *key_str;
    AC_GetRString(&ac, &key_str, 0);
    model->infokey = RAI_HoldString(NULL, key_str);

    // Parse <backend> argument.
    const char *backend_str;
    int backend;
    AC_GetString(&ac, &backend_str, NULL, 0);
    if (strcasecmp(backend_str, "TF") == 0) {
        backend = RAI_BACKEND_TENSORFLOW;
    } else if (strcasecmp(backend_str, "TFLITE") == 0) {
        backend = RAI_BACKEND_TFLITE;
    } else if (strcasecmp(backend_str, "TORCH") == 0) {
        backend = RAI_BACKEND_TORCH;
    } else if (strcasecmp(backend_str, "ONNX") == 0) {
        backend = RAI_BACKEND_ONNXRUNTIME;
    } else {
        RAI_SetError(err, RAI_EMODELCREATE, "ERR unsupported backend");
        return REDISMODULE_ERR;
    }
    model->backend = backend;

    // Parse <backend> argument: check that the device string is "CPU", "GPU" or
    // "GPU:<n>" where <n> is a number (contains digits only).
    const char *device_str;
    AC_GetString(&ac, &device_str, NULL, 0);
    bool valid_device = false;
    if (strcasecmp(device_str, "CPU") == 0 || strcasecmp(device_str, "GPU") == 0) {
        valid_device = true;
    } else if (strncasecmp(device_str, "GPU:", 4) == 0 && strlen(device_str) <= 10) {
        bool digits_only = true;
        for (size_t i = 5; i < strlen(device_str); i++) {
            if (device_str[i] < '0' || device_str[i] > '9') {
                digits_only = false;
                break;
            }
        }
        valid_device = digits_only;
    }
    if (!valid_device) {
        RAI_SetError(err, RAI_EMODELCREATE, "ERR Invalid DEVICE");
        return REDISMODULE_ERR;
    }
    model->devicestr = RedisModule_Strdup(device_str);

    // Parse <tag> argument, and add model key to the stats dict.
    RedisModuleString *tag = NULL;
    if (AC_AdvanceIfMatch(&ac, "TAG")) {
        AC_GetRString(&ac, &tag, 0);
    }
    if (tag) {
        model->tag = RAI_HoldString(NULL, tag);
    } else {
        model->tag = RedisModule_CreateString(NULL, "", 0);
    }

    // Parse the optional args of BATCHSIZE, MINBATCHSIZE and MINBATCHTIMEOUT, and set model opts.
    RAI_ModelOpts opts;
    if (_ModelSetCommand_ParseBatchingArgs(&ac, &opts, backend, err) != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    opts.backends_intra_op_parallelism = getBackendsIntraOpParallelism();
    opts.backends_inter_op_parallelism = getBackendsInterOpParallelism();
    model->opts = opts;

    // Parse inputs and output names (this arguments are mandatory only for TF models)
    // and store them in model objects.
    if (backend == RAI_BACKEND_TENSORFLOW) {
        if (_ModelSetCommand_ParseIOArgs(model, &ac, backend, err) != REDISMODULE_OK) {
            return REDISMODULE_ERR;
        }
    }

    // Parse model blob (final argument), and store it in the model.
    const char *blob_matches[1] = {"BLOB"};
    AC_AdvanceUntilMatches(&ac, 1, blob_matches);
    if (AC_IsAtEnd(&ac)) {
        RAI_SetError(err, RAI_EMODELCREATE, "ERR Insufficient arguments, missing model BLOB");
        return REDISMODULE_ERR;
    }
    AC_Advance(&ac);
    _ModelSetCommand_ParseBlob(model, &ac);
    return REDISMODULE_OK;
}

int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag) {

    int flags = RedisModule_GetContextFlags(ctx);
    bool blocking_not_allowed = (flags & (REDISMODULE_CTX_FLAGS_MULTI | REDISMODULE_CTX_FLAGS_LUA));
    if (blocking_not_allowed)
        return RedisModule_ReplyWithError(
            ctx, "ERR Cannot run RedisAI command within a transaction or a LUA script");

    RedisAI_RunInfo *rinfo;
    RAI_InitRunInfo(&rinfo);
    int status = REDISMODULE_ERR;

    switch (command) {
    case CMD_MODELRUN:
        rinfo->single_op_dag = 1;
        RAI_DagOp *modelRunOp;
        RAI_InitDagOp(&modelRunOp);
        rinfo->dagOps = array_append(rinfo->dagOps, modelRunOp);
        status = ParseModelRunCommand(rinfo, modelRunOp, argv, argc);
        break;
    case CMD_SCRIPTRUN:
        rinfo->single_op_dag = 1;
        RAI_DagOp *scriptRunOp;
        RAI_InitDagOp(&scriptRunOp);
        rinfo->dagOps = array_append(rinfo->dagOps, scriptRunOp);
        status = ParseScriptRunCommand(rinfo, scriptRunOp, argv, argc);
        break;
    case CMD_DAGRUN:
        status = ParseDAGRunCommand(rinfo, ctx, argv, argc, ro_dag);
        break;
    default:
        break;
    }
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_OK;
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);

    rinfo->OnFinish = DAG_ReplyAndUnblock;
    rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, RunInfo_FreeData, 0);
    if (DAG_InsertDAGToQueue(rinfo) != REDISMODULE_OK) {
        RedisModule_UnblockClient(rinfo->client, rinfo);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}
