
#include "command_parser.h"
#include "redismodule.h"
#include "run_info.h"
#include "modelRun_ctx.h"
#include "DAG/dag.h"
#include "DAG/dag_parser.h"
#include "util/string_utils.h"

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
