#include "script_commands_parser.h"

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

static bool _Script_buildInputsBySchema(RAI_ScriptRunCtx *sctx, RedisModuleString** inputs, RedisModuleString ***inkeys, RAI_Error *error) {
    int signatureListCount = 0;
    size_t nlists;
    
    TorchScriptFunctionArgumentType *signature =  RAI_ScriptRunCtxGetSignature(sctx);
    if(!signature) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "Wrong number of inputs provided to AI.SCRIPTEXECUTE command");
        return false;
    }
    size_t nArguments = array_len(sctx);
    size_t nInputs = array_len(inputs);
    size_t inputsIdx = 0;
    size_t listIdx = 0;

    if(nInputs < nArguments) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "Wrong number of inputs provided to AI.SCRIPTEXECUTE command");
        return false;
    }
    for (size_t i = 0; i < nArguments; i++) {
        if(signature[i] == UNKOWN) {
            RAI_SetError(error, RAI_ESCRIPTRUN, "Unsupported argument type in AI.SCRIPTEXECUTE command");
            return false;
        }
        // Collect the inputs tensor names from the current list
        if(signature[i] == LIST) {
            signatureListCount++;
            size_t listLen =  RAI_ScriptRunCtxGetInputListLen(sctx, listIdx);
            for(int j = 0; j < listLen; j++ ){
                *inkeys = array_append(inkeys, inputs[inputsIdx++]);
            }
            continue;
        }
        if(signature[i] != TENSOR) {
            // Input is not a tensor. It is string/int/float, so it is not required a key.
            continue;
        }
        else {
            // Input is a tensor, add its name to the inkeys.
            *inkeys = array_append(inkeys, inputs[inputsIdx++]);
        }
    } 
    if(signatureListCount != nlists) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
        return false;
    }

    return true;
}

static int _ScriptExecuteCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           RAI_Error *error, RedisModuleString ***inkeys,
                                           RedisModuleString ***outkeys, RAI_ScriptRunCtx* sctx, long long *timeout) {
    int argpos = 3;
    int inputs_mask = 1 << 0;
    int outputs_mask = 1 << 1;
    int keys_mask = 1 << 2;
    // Local input context to verify correctness.
    array_new_on_stack(RedisModuleString*, 10, inputs);
    int mask = 0;

    while (argpos < argc ) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);

        // Parse timeout arg if given and store it in timeout.
        if (!strcasecmp(arg_string, "TIMEOUT")) {
            argpos++;
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR No value provided for TIMEOUT in AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            if (ParseTimeout(argv[argpos], error, timeout) == REDISMODULE_ERR)
                goto cleanup;
            // No other arguments expected after timeout.
            break;
        }

        if (!strcasecmp(arg_string, "INPUTS")) {
            // Check for already given inputs.
            if(mask & inputs_mask) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered INPUTS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            // Update mask.
            mask |= inputs_mask;
            // Read input number.
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for input count in AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            // Check validity of input numbers.
            argpos++;
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                goto cleanup;
            }
            // Add to local input context.
            for (; argpos < first_input_pos + ninputs; argpos++) {
                inputs = array_append(inputs, RAI_HoldString(NULL, argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "KEYS")) {
            // Check for already given keys.
            if(mask & keys_mask) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered KEYS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            // Update mask.
            mask |= keys_mask;
            // Read key number.
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for key count in AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            // Check validity of key numbers.
            argpos++;
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                goto cleanup;
            }
            // Verify given keys in local shard.
            for (; argpos < first_input_pos + ninputs; argpos++) {
                if(!VerifyKeyInThisShard(ctx, argv[argpos])) {
                    RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR CROSSSLOT Keys in request don't hash to the same slot");
                    goto cleanup;
                }
            }
            continue;
        }
        if (!strcasecmp(arg_string, "OUTPUTS")) {
            // Check for already given outputs.
            if(mask & keys_mask) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered KEYS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            // Update mask.
            mask |= outputs_mask;
            // Read output number.
            argpos++;
            long long noutputs;
            if (RedisModule_StringToLongLong(argv[argpos], &noutputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for output count in AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            // Check validity of output numbers.
            argpos++;
            size_t first_output_pos = argpos;
            if (first_output_pos + noutputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of output keys to AI.SCRIPTEXECUTE command does not match "
                             "the number of given arguments");
                goto cleanup;
            }
            for (; argpos < first_output_pos + noutputs; argpos++) {
                *outkeys = array_append(*outkeys, RAI_HoldString(NULL, argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "LIST_INPUTS")) {
            // Read list size.
            argpos++;
            long long ninputs;
            if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid argument for list input count in AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            // Check validity of current list size.
            argpos++;
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of list input keys to AI.SCRIPTEXECUTE command does not "
                             "match the number of given arguments");
                goto cleanup;
            }
            for (; argpos < first_input_pos + ninputs; argpos++) {
                inputs = array_append(inputs, RAI_HoldString(NULL, argv[argpos]));
            }
            RAI_ScriptRunCtxAddListSize(sctx, ninputs);
            continue;
        }

        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Unrecongnized parameter to AI.SCRIPTEXECUTE");
        goto cleanup;
    }
    if (argpos != argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Encountered problem parsing AI.SCRIPTEXECUTE");
        return REDISMODULE_ERR;
    }

    if(!_Script_buildInputsBySchema(sctx, inputs, inkeys, error)) {
        goto cleanup;
    }
    array_free(inputs);

    return REDISMODULE_OK;
cleanup:
    array_free(inputs);
    return REDISMODULE_ERR;
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
                                        &currentOp->outkeys, sctx, &timeout) == REDISMODULE_ERR) {
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