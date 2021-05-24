#include "script_commands_parser.h"
#include "parse_utils.h"
#include "execution/utils.h"
#include "util/string_utils.h"
#include "execution/execution_contexts/scriptRun_ctx.h"

static bool _Script_buildInputsBySchema(RAI_ScriptRunCtx *sctx, RedisModuleString **inputs,
                                        RedisModuleString ***inkeys, RAI_Error *error) {
    int signatureListCount = 0;

    TorchScriptFunctionArgumentType *signature = RAI_ScriptRunCtxGetSignature(sctx);
    if (!signature) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "Wrong function name given to AI.SCRIPTEXECUTE command");
        return false;
    }
    size_t nlists = array_len(sctx->listSizes);
    size_t nArguments = array_len(signature);
    size_t nInputs = array_len(inputs);
    size_t inputsIdx = 0;
    size_t listIdx = 0;

    if (nInputs < nArguments) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "Wrong number of inputs provided to AI.SCRIPTEXECUTE command");
        return false;
    }
    for (size_t i = 0; i < nArguments; i++) {
        switch (signature[i]) {
        case UNKOWN: {
            RAI_SetError(error, RAI_ESCRIPTRUN,
                         "Unsupported argument type in AI.SCRIPTEXECUTE command");
            return false;
        }
        case TENSOR_LIST: {
            // Collect the inputs tensor names from the current list
            signatureListCount++;
            if (signatureListCount > nlists) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
                return false;
            }
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx++);
            for (size_t j = 0; j < listLen; j++) {
                *inkeys = array_append(*inkeys, RAI_HoldString(inputs[inputsIdx++]));
            }
            break;
        }
        case INT_LIST: {
            signatureListCount++;
            if (signatureListCount > nlists) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
                return false;
            }
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx++);
            for (size_t j = 0; j < listLen; j++) {
                long long l;
                RedisModule_StringToLongLong(inputs[inputsIdx++], &l);
                RAI_ScriptRunCtxAddIntInput(sctx, (int32_t)l);
            }
            break;
        }
        case FLOAT_LIST: {
            signatureListCount++;
            if (signatureListCount > nlists) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
                return false;
            }
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx++);
            for (size_t j = 0; j < listLen; j++) {
                double d;
                RedisModule_StringToDouble(inputs[inputsIdx++], &d);
                RAI_ScriptRunCtxAddFloatInput(sctx, (float)d);
            }
            break;
        }
        case STRING_LIST: {
            signatureListCount++;
            if (signatureListCount > nlists) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
                return false;
            }
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx++);
            for (size_t j = 0; j < listLen; j++) {
                RAI_ScriptRunCtxAddRStringInput(sctx, inputs[inputsIdx++]);
            }
            break;
        }
        case INT: {
            long long l;
            RedisModule_StringToLongLong(inputs[inputsIdx++], &l);
            RAI_ScriptRunCtxAddIntInput(sctx, (int32_t)l);
            break;
        }
        case FLOAT: {
            double d;
            RedisModule_StringToDouble(inputs[inputsIdx++], &d);
            RAI_ScriptRunCtxAddFloatInput(sctx, (float)d);
            break;
        }
        case STRING: {
            // Input is a string.
            RAI_ScriptRunCtxAddRStringInput(sctx, inputs[inputsIdx++]);
            break;
        }
        case TENSOR:
        default: {
            // Input is a tensor, add its name to the inkeys.
            *inkeys = array_append(*inkeys, RAI_HoldString(inputs[inputsIdx++]));
            break;
        }
        }
    }
    if (signatureListCount != nlists) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
        return false;
    }

    return true;
}

static int _ScriptExecuteCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           RAI_Error *error, RedisModuleString ***inkeys,
                                           RedisModuleString ***outkeys, RAI_ScriptRunCtx *sctx,
                                           long long *timeout, bool keysRequired) {
    int argpos = 3;
    bool inputsDone = false;
    bool outputsDone = false;
    // Local input context to verify correctness.
    array_new_on_stack(RedisModuleString *, 10, inputs);

    if (keysRequired) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "KEYS")) {
            const int parse_result = ParseKeysArgs(ctx, &argv[argpos], argc - argpos, error);
            if (parse_result <= 0)
                return REDISMODULE_ERR;
            argpos += parse_result;
        }
        // argv[3] is not KEYS in AI.SCRIPTEXECUTE command (i.e., not in a DAG).
        else {
            RAI_SetError(error, RAI_ESCRIPTRUN,
                         "ERR KEYS scope must be provided first for AI.SCRIPTEXECUTE command");
            goto cleanup;
        }
    }

    while (argpos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        // See that no additional KEYS scope is provided.
        if (!strcasecmp(arg_string, "KEYS")) {
            RAI_SetError(error, RAI_ESCRIPTRUN,
                         "ERR Already encountered KEYS scope in current command");
            goto cleanup;
        }
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
            if (inputsDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered INPUTS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            inputsDone = true;
            // Read input number.
            argpos++;
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
                goto cleanup;
            }
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
                inputs = array_append(inputs, RAI_HoldString(argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "OUTPUTS")) {
            // Check for already given outputs.
            if (outputsDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered OUTPUTS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            // Update mask.
            outputsDone = true;
            // Read output number.
            argpos++;
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
                goto cleanup;
            }
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
                *outkeys = array_append(*outkeys, RAI_HoldString(argv[argpos]));
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
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
                goto cleanup;
            }
            size_t first_input_pos = argpos;
            if (first_input_pos + ninputs > argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR number of list input keys to AI.SCRIPTEXECUTE command does not "
                             "match the number of given arguments");
                goto cleanup;
            }
            for (; argpos < first_input_pos + ninputs; argpos++) {
                inputs = array_append(inputs, RAI_HoldString(argv[argpos]));
            }
            RAI_ScriptRunCtxAddListSize(sctx, ninputs);
            continue;
        }
        size_t error_len = strlen("ERR Invalid AI.SCRIPTEXECUTE command. Unexpected argument: ") +
                           strlen(arg_string) + 1;
        char error_str[error_len];
        sprintf(error_str, "ERR Invalid AI.SCRIPTEXECUTE command. Unexpected argument: %s",
                arg_string);
        RAI_SetError(error, RAI_ESCRIPTRUN, error_str);
        goto cleanup;
    }
    if (argpos != argc) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Encountered problem parsing AI.SCRIPTEXECUTE");
        goto cleanup;
    }

    if (!_Script_buildInputsBySchema(sctx, inputs, inkeys, error)) {
        goto cleanup;
    }
    for (size_t i = 0; i < array_len(inputs); i++) {
        RedisModule_FreeString(ctx, inputs[i]);
    }
    array_free(inputs);

    return REDISMODULE_OK;
cleanup:
    for (size_t i = 0; i < array_len(inputs); i++) {
        RedisModule_FreeString(ctx, inputs[i]);
    }
    array_free(inputs);
    return REDISMODULE_ERR;
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

    RAI_DagOpSetRunKey(currentOp, RAI_HoldString(argv[1]));

    const char *func_name = ScriptCommand_GetFunctionName(argv[2]);
    if (!func_name) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN, "ERR function name not specified");
        goto cleanup;
    }

    sctx = RAI_ScriptRunCtxCreate(script, func_name);
    long long timeout = 0;
    if (_ScriptExecuteCommand_ParseArgs(ctx, argv, argc, error, &currentOp->inkeys,
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
    currentOp->ectx = (RAI_ExecutionCtx*)sctx;
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
