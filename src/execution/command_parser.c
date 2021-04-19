#include "redismodule.h"
#include "run_info.h"
#include "command_parser.h"
#include "DAG/dag.h"
#include "DAG/dag_op.h"
#include "DAG/dag_parser.h"
#include "util/string_utils.h"
#include "execution/modelRun_ctx.h"
#include "deprecated.h"
#include "utils.h"

static int _ModelExecuteCommand_ParseArgs(RedisModuleCtx *ctx, int argc, RedisModuleString **argv,
                                          RAI_Model **model, RAI_Error *error,
                                          RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                          RedisModuleString **runkey, long long *timeout) {

    if (argc < 8) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR wrong number of arguments for 'AI.MODELEXECUTE' command");
        return REDISMODULE_ERR;
    }
    size_t arg_pos = 1;
    const int status = RAI_GetModelFromKeyspace(ctx, argv[arg_pos], model, REDISMODULE_READ, error);
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    *runkey = RAI_HoldString(NULL, argv[arg_pos++]);
    const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos++], NULL);

    if (strcasecmp(arg_string, "INPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
        return REDISMODULE_ERR;
    }

    long long ninputs = 0, noutputs = 0;
    if (RedisModule_StringToLongLong(argv[arg_pos++], &ninputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid argument for input_count");
        return REDISMODULE_ERR;
    }
    if (ninputs <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Input count must be a positive integer");
        return REDISMODULE_ERR;
    }
    if ((*model)->ninputs != ninputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as INPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }
    // arg_pos = 4
    size_t first_input_pos = arg_pos;
    if (first_input_pos + ninputs > argc) {
        RAI_SetError(
            error, RAI_EMODELRUN,
            "ERR number of input keys to AI.MODELEXECUTE command does not match the number of "
            "given arguments");
        return REDISMODULE_ERR;
    }
    for (; arg_pos < first_input_pos + ninputs; arg_pos++) {
        *inkeys = array_append(*inkeys, RAI_HoldString(NULL, argv[arg_pos]));
    }

    if (argc == arg_pos ||
        strcasecmp(RedisModule_StringPtrLen(argv[arg_pos++], NULL), "OUTPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR OUTPUTS not specified");
        return REDISMODULE_ERR;
    }
    if (argc == arg_pos ||
        RedisModule_StringToLongLong(argv[arg_pos++], &noutputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid argument for output_count");
    }
    if (noutputs <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Output count must be a positive integer");
        return REDISMODULE_ERR;
    }
    if ((*model)->noutputs != noutputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as OUTPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }
    // arg_pos = ninputs+6, the argument that we already parsed are:
    // AI.MODELEXECUTE <model_key> INPUTS <input_count> <input> ... OUTPUTS <output_count>
    size_t first_output_pos = arg_pos;
    if (first_output_pos + noutputs > argc) {
        RAI_SetError(
            error, RAI_EMODELRUN,
            "ERR number of output keys to AI.MODELEXECUTE command does not match the number of "
            "given arguments");
        return REDISMODULE_ERR;
    }
    for (; arg_pos < first_output_pos + noutputs; arg_pos++) {
        *outkeys = array_append(*outkeys, RAI_HoldString(NULL, argv[arg_pos]));
    }
    if (arg_pos == argc) {
        return REDISMODULE_OK;
    }

    // Parse timeout arg if given and store it in timeout.
    char *error_str;
    arg_string = RedisModule_StringPtrLen(argv[arg_pos++], NULL);
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (arg_pos == argc) {
            RAI_SetError(error, RAI_EMODELRUN, "ERR No value provided for TIMEOUT");
            return REDISMODULE_ERR;
        }
        if (ParseTimeout(argv[arg_pos++], error, timeout) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
    } else {
        error_str = RedisModule_Alloc(strlen("Invalid argument: ") + strlen(arg_string) + 1);
        sprintf(error_str, "Invalid argument: %s", arg_string);
        RAI_SetError(error, RAI_EMODELRUN, error_str);
        RedisModule_Free(error_str);
        return REDISMODULE_ERR;
    }

    // There are no more valid args to be processed.
    if (arg_pos != argc) {
        arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        error_str = RedisModule_Alloc(strlen("Invalid argument: ") + strlen(arg_string) + 1);
        sprintf(error_str, "Invalid argument: %s", arg_string);
        RAI_SetError(error, RAI_EMODELRUN, error_str);
        RedisModule_Free(error_str);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                          RedisModuleString **outkeys, RAI_ModelRunCtx *mctx, RAI_Error *err) {

    RAI_Model *model = mctx->model;
    RAI_Tensor *t;
    RedisModuleKey *key;
    char *opname = NULL;
    size_t ninputs = array_len(inkeys), noutputs = array_len(outkeys);
    for (size_t i = 0; i < ninputs; i++) {
        const int status =
            RAI_GetTensorFromKeyspace(ctx, inkeys[i], &key, &t, REDISMODULE_READ, err);
        if (status == REDISMODULE_ERR) {
            return REDISMODULE_ERR;
        }
        if (model->inputs)
            opname = model->inputs[i];
        RAI_ModelRunCtxAddInput(mctx, opname, t);
    }

    for (size_t i = 0; i < noutputs; i++) {
        if (model->outputs) {
            opname = model->outputs[i];
        }
        if (!VerifyKeyInThisShard(ctx, outkeys[i])) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EMODELRUN,
                         "ERR CROSSSLOT Keys in request don't hash to the same slot");
            return REDISMODULE_ERR;
        }
        RAI_ModelRunCtxAddOutput(mctx, opname);
    }
    return REDISMODULE_OK;
}

int ParseModelExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                             int argc) {

    int res = REDISMODULE_ERR;
    // Build a ModelRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RAI_Model *model;
    long long timeout = 0;
    if (_ModelExecuteCommand_ParseArgs(ctx, argc, argv, &model, rinfo->err, &currentOp->inkeys,
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
        if (ModelRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, mctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}

static int _ScriptRunCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                       RAI_Error *error, RedisModuleString ***inkeys,
                                       RedisModuleString ***outkeys, long long *timeout,
                                       size_t **listSizes) {

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
    if (varidic_start_pos != -1) {
        *listSizes = array_append(*listSizes, ninputs - varidic_start_pos);
    }

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


int ParseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout) {

    const int retval = RedisModule_StringToLongLong(timeout_arg, timeout);
    if (retval != REDISMODULE_OK || *timeout <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static RAI_Script *_ScriptCommand_GetScript(RedisModuleCtx *ctx, RedisModuleString *scriptName,
                                            RAI_Error *error) {
    RAI_Script *script = NULL;
    RAI_GetScriptFromKeyspace(ctx, scriptName, &script, REDISMODULE_READ, error);
    return script;
}

static const char *_ScriptCommand_GetFunction(RedisModuleString *functionName) {
    const char *functionName_cstr = RedisModule_StringPtrLen(functionName, NULL);
    if (!strcasecmp(functionName_cstr, "TIMEOUT") || !strcasecmp(functionName_cstr, "INPUTS") ||
        !strcasecmp(functionName_cstr, "OUTPUTS")) {
        return NULL;
    }
    return functionName_cstr;

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

    RAI_Script *script = _ScriptCommand_GetScript(ctx, argv[1], rinfo->err);
    if (!script) {
        goto cleanup;
    }

    RAI_DagOpSetRunKey(currentOp, RAI_HoldString(ctx, argv[1]));

    const char *func_name = _ScriptCommand_GetFunction(argv[2]);
    if (!func_name) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN, "ERR function name not specified");
        goto cleanup;
    }

    RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(script, func_name);
    long long timeout = 0;
    if (_ScriptRunCommand_ParseArgs(ctx, argv, argc, rinfo->err, &currentOp->inkeys,
                                    &currentOp->outkeys, &timeout,
                                    &sctx->listSizes) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }
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
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}

static int _ScriptExecuteCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           RAI_Error *error, RedisModuleString ***inkeys,
                                           RedisModuleString ***outkeys, long long *timeout,
                                           size_t **listSizes) {
    bool timeout_set = false;
    int argpos;
    for (argpos = 3; argpos < argc; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);

        // Parse timeout arg if given and store it in timeout
        if (!strcasecmp(arg_string, "TIMEOUT")) {
            if (timeout_set) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an TIMEOUT section in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            argpos++;
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR No value provided for TIMEOUT in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            if (_parseTimeout(argv[argpos], error, timeout) == REDISMODULE_ERR)
                return REDISMODULE_ERR;
            break;
        }

        if (!strcasecmp(arg_string, "INPUTS")) {
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for input count in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                return REDISMODULE_ERR;
            }
            for (; argpos < first_input_pos + ninputs; argpos++) {
                *inkeys = array_append(*inkeys, RAI_HoldString(NULL, argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "KEYS")) {
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for key count in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                return REDISMODULE_ERR;
            }
            for (; argpos < first_input_pos + ninputs; argpos++) {
                // Do nothing.
            }
            continue;
        }
        if (!strcasecmp(arg_string, "OUTPUTS")) {
            if (array_len(outkeys) != 0) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already encountered an OUTPUTS keyword in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            argpos++;
            long long noutputs;
            if (RedisModule_StringToLongLong(argv[argpos], &noutputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for output count in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            size_t first_output_pos = argpos;
            if (first_output_pos + noutputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of output keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                return REDISMODULE_ERR;
            }
            for (; argpos < first_output_pos + noutputs; argpos++) {
                *outkeys = array_append(*outkeys, RAI_HoldString(NULL, argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "LIST_INPUTS")) {
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for list input count in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of list input keys to AI.SCRIPTEXECUTE command does not "
                             "match the number of given arguments");
                return REDISMODULE_ERR;
            }
            for (; argpos < first_input_pos + ninputs; argpos++) {
                *inkeys = array_append(*inkeys, RAI_HoldString(NULL, argv[argpos]));
            }
            *listSizes = array_append(*listSizes, ninputs);
            continue;
        }

        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Unrecongnized parameter to AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    if (argpos != argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Encountered problem parsing AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}

int ParseScriptExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp,
                              RedisModuleString **argv, int argc) {

    if (argc < 3) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN,
                     "ERR wrong number of arguments for 'AI.SCRIPTEXECUTE' command");
        return REDISMODULE_ERR;
    }

    int res = REDISMODULE_ERR;
    // Build a ScriptRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);

    RAI_Script *script = _ScriptCommand_GetScript(ctx, argv[1], rinfo->err);
    if (!script) {
        goto cleanup;
    }

    RAI_DagOpSetRunKey(currentOp, RAI_HoldString(ctx, argv[1]));

    const char *func_name = _ScriptCommand_GetFunction(argv[2]);
    if (!func_name) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN, "ERR function name not specified");
        goto cleanup;
    }

    RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(script, func_name);
    long long timeout = 0;
    size_t *listSizes = array_new(size_t, 1);
    if (_ScriptExecuteCommand_ParseArgs(ctx, argv, argc, rinfo->err, &currentOp->inkeys,
                                        &currentOp->outkeys, &timeout,
                                        &sctx->listSizes) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }
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
    case CMD_SCRIPTEXECUTE:
        rinfo->single_op_dag = 1;
        RAI_DagOp *scriptExecOp;
        RAI_InitDagOp(&scriptExecOp);
        rinfo->dagOps = array_append(rinfo->dagOps, scriptExecOp);
        status = ParseScriptExecuteCommand(rinfo, scriptExecOp, argv, argc);
        break;
    case CMD_DAGRUN:
        status = ParseDAGRunCommand(rinfo, ctx, argv, argc, ro_dag);
        break;
    case CMD_MODELEXECUTE:
        rinfo->single_op_dag = 1;
        RAI_DagOp *modelExecuteOp;
        RAI_InitDagOp(&modelExecuteOp);
        rinfo->dagOps = array_append(rinfo->dagOps, modelExecuteOp);
        status = ParseModelExecuteCommand(rinfo, modelExecuteOp, argv, argc);
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
