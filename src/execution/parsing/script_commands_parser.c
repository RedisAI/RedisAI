/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "script_commands_parser.h"
#include "parse_utils.h"
#include "execution/utils.h"
#include "util/string_utils.h"
#include "execution/execution_contexts/scriptRun_ctx.h"

static int _ScriptExecuteCommand_ParseKeys(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           size_t *argpos, RAI_Error *error,
                                           RAI_ScriptRunCtx *sctx) {
    size_t localArgpos = *argpos;
    const int keys_validation_result =
        ValidateKeysArgs(ctx, &argv[localArgpos - 1], argc - (localArgpos - 1), error);
    if (keys_validation_result <= 0)
        return REDISMODULE_ERR;

    if (localArgpos >= argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    long long nKeys;
    if (RedisModule_StringToLongLong(argv[localArgpos], &nKeys) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR Invalid argument for keys count in AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    // Check validity of keys numbers.
    localArgpos++;
    size_t firstKeyPos = localArgpos;
    if (firstKeyPos + nKeys > argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR number of keys to AI.SCRIPTEXECUTE command does not match "
                     "the number of given arguments");
        return REDISMODULE_ERR;
    }
    // Add to script run context.
    for (; localArgpos < firstKeyPos + nKeys; localArgpos++) {
        RAI_ScriptRunCtxAddKeyInput(sctx, RAI_HoldString(argv[localArgpos]));
    }
    *argpos = localArgpos;
    return REDISMODULE_OK;
}

static int _ScriptExecuteCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           size_t *argpos, RAI_Error *error,
                                           RAI_ScriptRunCtx *sctx) {
    size_t localArgpos = *argpos;
    if (localArgpos >= argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    long long nArgs;
    if (RedisModule_StringToLongLong(argv[localArgpos], &nArgs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR Invalid argument for keys count in AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    // Check validity of args numbers.
    localArgpos++;
    size_t firstArgPos = localArgpos;
    if (firstArgPos + nArgs > argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR number of args to AI.SCRIPTEXECUTE command does not match "
                     "the number of given arguments");
        return REDISMODULE_ERR;
    }
    // Add to script run context.
    for (; localArgpos < firstArgPos + nArgs; localArgpos++) {
        RAI_ScriptRunCtxAddArgInput(sctx, RAI_HoldString(argv[localArgpos]));
    }
    *argpos = localArgpos;
    return REDISMODULE_OK;
}

static int _ScriptExecuteCommand_ParseInputs(RedisModuleCtx *ctx, RedisModuleString **argv,
                                             int argc, size_t *argpos, RAI_Error *error,
                                             RedisModuleString ***inputs) {
    size_t localArgpos = *argpos;
    if (localArgpos >= argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    long long nInputs;
    if (RedisModule_StringToLongLong(argv[localArgpos], &nInputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR Invalid argument for inputs count in AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    // Check validity of inputs numbers.
    localArgpos++;
    size_t firstInputPos = localArgpos;
    if (firstInputPos + nInputs > argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR number of inputs to AI.SCRIPTEXECUTE command does not match "
                     "the number of given arguments");
        return REDISMODULE_ERR;
    }
    // Add to inputs array.
    for (; localArgpos < firstInputPos + nInputs; localArgpos++) {
        *inputs = array_append(*inputs, RAI_HoldString(argv[localArgpos]));
    }
    *argpos = localArgpos;
    return REDISMODULE_OK;
}

static int _ScriptExecuteCommand_ParseOutputs(RedisModuleCtx *ctx, RedisModuleString **argv,
                                              int argc, size_t *argpos, RAI_Error *error,
                                              RedisModuleString ***outputs) {
    size_t localArgpos = *argpos;
    if (localArgpos >= argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    long long nOutputs;
    if (RedisModule_StringToLongLong(argv[localArgpos], &nOutputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR Invalid argument for outputs count in AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }
    // Check validity of outputs numbers.
    localArgpos++;
    size_t firstOutputPos = localArgpos;
    if (firstOutputPos + nOutputs > argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR number of outputs to AI.SCRIPTEXECUTE command does not match "
                     "the number of given arguments");
        return REDISMODULE_ERR;
    }
    // Add to outputs array.
    for (; localArgpos < firstOutputPos + nOutputs; localArgpos++) {
        *outputs = array_append(*outputs, RAI_HoldString(argv[localArgpos]));
    }
    *argpos = localArgpos;
    return REDISMODULE_OK;
}

static int _ScriptExecuteCommand_ParseCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                              int argc, RAI_Error *error,
                                              RedisModuleString ***inputs,
                                              RedisModuleString ***outputs, RAI_ScriptRunCtx *sctx,
                                              long long *timeout, bool keysRequired) {
    size_t argpos = 3;
    bool inputsDone = false;
    bool outputsDone = false;
    bool keysDone = false;
    bool argsDone = false;

    if (keysRequired) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "KEYS")) {
            argpos++;
            keysDone = true;
            if (_ScriptExecuteCommand_ParseKeys(ctx, argv, argc, &argpos, error, sctx) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
        } else if (!strcasecmp(arg_string, "INPUTS")) {
            argpos++;
            inputsDone = true;
            if (_ScriptExecuteCommand_ParseInputs(ctx, argv, argc, &argpos, error, inputs) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
        }
        // argv[3] is not KEYS or INPUTS in AI.SCRIPTEXECUTE command (i.e., not in a DAG).
        else {
            RAI_SetError(
                error, RAI_ESCRIPTRUN,
                "ERR KEYS or INPUTS scope must be provided first for AI.SCRIPTEXECUTE command");
            return REDISMODULE_ERR;
        }
    }

    while (argpos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos++], NULL);
        // Parse timeout arg if given and store it in timeout.
        if (!strcasecmp(arg_string, "TIMEOUT")) {
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR No value provided for TIMEOUT in AI.SCRIPTEXECUTE");
                return REDISMODULE_ERR;
            }
            if (ParseTimeout(argv[argpos++], error, timeout) == REDISMODULE_ERR)
                return REDISMODULE_ERR;
            // No other arguments expected after timeout.
            break;
        }

        if (!strcasecmp(arg_string, "KEYS")) {
            if (keysDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered KEYS scope in AI.SCRIPTEXECUTE command");
                return REDISMODULE_ERR;
            }
            keysDone = true;
            if (_ScriptExecuteCommand_ParseKeys(ctx, argv, argc, &argpos, error, sctx) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
            continue;
        }

        if (!strcasecmp(arg_string, "ARGS")) {
            if (argsDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered ARGS scope in AI.SCRIPTEXECUTE command");
                return REDISMODULE_ERR;
            }
            argsDone = true;
            if (_ScriptExecuteCommand_ParseArgs(ctx, argv, argc, &argpos, error, sctx) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
            continue;
        }

        if (!strcasecmp(arg_string, "INPUTS")) {
            // Check for already given inputs.
            if (inputsDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered INPUTS scope in AI.SCRIPTEXECUTE command");
                return REDISMODULE_ERR;
            }
            inputsDone = true;
            if (_ScriptExecuteCommand_ParseInputs(ctx, argv, argc, &argpos, error, inputs) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "OUTPUTS")) {
            // Check for already given outputs.
            if (outputsDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered OUTPUTS scope in AI.SCRIPTEXECUTE command");
                return REDISMODULE_ERR;
            }
            outputsDone = true;
            if (_ScriptExecuteCommand_ParseOutputs(ctx, argv, argc, &argpos, error, outputs) ==
                REDISMODULE_ERR) {
                return REDISMODULE_ERR;
            }
            continue;
        }

        size_t error_len = strlen("ERR Invalid AI.SCRIPTEXECUTE command. Unexpected argument: ") +
                           strlen(arg_string) + 1;
        char error_str[error_len];
        sprintf(error_str, "ERR Invalid AI.SCRIPTEXECUTE command. Unexpected argument: %s",
                arg_string);
        RAI_SetError(error, RAI_ESCRIPTRUN, error_str);
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

    RAI_Error *error = rinfo->err;
    if (argc < 3) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR wrong number of arguments for 'AI.SCRIPTEXECUTE' command");
        return REDISMODULE_ERR;
    }

    int res = REDISMODULE_ERR;
    // Build a ScriptRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);

    RAI_Script *script = NULL;
    RAI_ScriptRunCtx *sctx = NULL;
    RedisModuleString *scriptName = argv[1];
    RAI_GetScriptFromKeyspace(ctx, scriptName, &script, REDISMODULE_READ, error);
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
    if (_ScriptExecuteCommand_ParseCommand(ctx, argv, argc, error, &currentOp->inkeys,
                                           &currentOp->outkeys, sctx, &timeout,
                                           rinfo->single_op_dag) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(error, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ScriptRunCtx, bring inputs from key space.
        if (ScriptRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, sctx, error) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;
    RedisModule_FreeThreadSafeContext(ctx);
    currentOp->ectx = (RAI_ExecutionCtx *)sctx;
    currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    currentOp->devicestr = sctx->script->devicestr;
    return res;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    if (sctx) {
        RAI_ScriptRunCtxFree(sctx);
    }
    return res;
}
