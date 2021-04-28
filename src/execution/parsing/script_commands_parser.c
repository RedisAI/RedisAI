#include "script_commands_parser.h"
#include "parse_utils.h"
#include "execution/utils.h"
#include "util/string_utils.h"
#include "execution/execution_contxets/scriptRun_ctx.h"

static bool _Script_buildInputsBySchema(RAI_ScriptRunCtx *sctx, RedisModuleString **inputs,
                                        RedisModuleString ***inkeys, RAI_Error *error) {
    int signatureListCount = 0;
    size_t nlists = array_len(sctx->listSizes);

    TorchScriptFunctionArgumentType *signature = RAI_ScriptRunCtxGetSignature(sctx);
    if (!signature) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "Wrong number of inputs provided to AI.SCRIPTEXECUTE command");
        return false;
    }
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
        if (signature[i] == UNKOWN) {
            RAI_SetError(error, RAI_ESCRIPTRUN,
                         "Unsupported argument type in AI.SCRIPTEXECUTE command");
            return false;
        }
        // Collect the inputs tensor names from the current list
        if (signature[i] == TENSOR_LIST) {
            signatureListCount++;
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx);
            for (int j = 0; j < listLen; j++) {
                *inkeys = array_append(*inkeys, inputs[inputsIdx++]);
            }
            continue;
        } else if (signature[i] == INT_LIST || signature[i] == FLOAT_LIST ||
                   signature[i] == STRING_LIST) {
            signatureListCount++;
            size_t listLen = RAI_ScriptRunCtxGetInputListLen(sctx, listIdx);
            for (int j = 0; j < listLen; j++) {
                // Input is not a tensor. It is string/int/float, so it is not required a key.
                sctx->otherInputs = array_append(sctx->otherInputs, inputs[inputsIdx++]);
            }
            continue;
        } else if (signature[i] != TENSOR) {
            // Input is not a tensor. It is string/int/float, so it is not required a key.
            sctx->otherInputs = array_append(sctx->otherInputs, inputs[inputsIdx++]);
            continue;
        } else {
            // Input is a tensor, add its name to the inkeys.
            *inkeys = array_append(*inkeys, inputs[inputsIdx++]);
        }
    }
    if (signatureListCount != nlists) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "Wrong number of lists provided in AI.SCRIPTEXECUTE command");
        return false;
    }

    return true;
}

// int ParseScriptExecuteArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
//                                            RAI_Error *error, RedisModuleString ***inkeys,
//                                            RedisModuleString ***outkeys, RAI_ScriptRunCtx *sctx,
//                                            long long *timeout) {
// int argpos = 0;
// bool inputsDone = false;
// bool outputsDone = false;
//     // Local input context to verify correctness.
//     array_new_on_stack(RedisModuleString *, 10, inputs);
// while (argpos < argc) {
//         const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);

//         // Parse timeout arg if given and store it in timeout.
//         if (!strcasecmp(arg_string, "TIMEOUT")) {
//             argpos++;
//             if (argpos >= argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR No value provided for TIMEOUT in AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             if (ParseTimeout(argv[argpos], error, timeout) == REDISMODULE_ERR)
//                 goto cleanup;
//             // No other arguments expected after timeout.
//             break;
//         }

//         if (!strcasecmp(arg_string, "INPUTS")) {
//             // Check for already given inputs.
//             if (inputsDone) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Already Encountered INPUTS scope in AI.SCRIPTEXECUTE command");
//                 goto cleanup;
//             }
//             inputsDone = true;
//             // Read input number.
//             argpos++;
//             if (argpos >= argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             long long ninputs;
//             if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid argument for input count in AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             // Check validity of input numbers.
//             argpos++;
//             size_t first_input_pos = argpos;
//             if (first_input_pos + ninputs > argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
//                              "the number of given arguments");
//                 goto cleanup;
//             }
//             // Add to local input context.
//             for (; argpos < first_input_pos + ninputs; argpos++) {
//                 inputs = array_append(inputs, RAI_HoldString(NULL, argv[argpos]));
//             }
//             continue;
//         }
//         if (!strcasecmp(arg_string, "KEYS")) {
//             // Check for already given keys.
//             if (KeysDone) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Already Encountered KEYS scope in AI.SCRIPTEXECUTE command");
//                 goto cleanup;
//             }
//             KeysDone = true;
//             // Read key number.
//             argpos++;
//             if (argpos >= argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             long long ninputs;
//             if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid argument for key count in AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             // Check validity of key numbers.
//             argpos++;
//             size_t first_input_pos = argpos;
//             if (first_input_pos + ninputs > argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR number of input keys to AI.SCRIPTEXECUTE command does not match "
//                              "the number of given arguments");
//                 goto cleanup;
//             }
//             // Verify given keys in local shard.
//             for (; argpos < first_input_pos + ninputs; argpos++) {
//                 if (!VerifyKeyInThisShard(ctx, argv[argpos])) {
//                     RAI_SetError(error, RAI_ESCRIPTRUN,
//                                  "ERR CROSSSLOT Keys in AI.SCRIPTEXECUTE request don't hash to the "
//                                  "same slot");
//                     goto cleanup;
//                 }
//             }
//             continue;
//         }
//         if (!strcasecmp(arg_string, "OUTPUTS")) {
//             // Check for already given outputs.
//             if (outputsDone) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Already Encountered OUTPUTS scope in AI.SCRIPTEXECUTE command");
//                 goto cleanup;
//             }
//             // Update mask.
//             outputsDone = true;
//             // Read output number.
//             argpos++;
//             if (argpos >= argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             long long noutputs;
//             if (RedisModule_StringToLongLong(argv[argpos], &noutputs) != REDISMODULE_OK) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid argument for output count in AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             // Check validity of output numbers.
//             argpos++;
//             size_t first_output_pos = argpos;
//             if (first_output_pos + noutputs > argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR number of output keys to AI.SCRIPTEXECUTE command does not match "
//                              "the number of given arguments");
//                 goto cleanup;
//             }
//             for (; argpos < first_output_pos + noutputs; argpos++) {
//                 *outkeys = array_append(*outkeys, RAI_HoldString(NULL, argv[argpos]));
//             }
//             continue;
//         }
//         if (!strcasecmp(arg_string, "LIST_INPUTS")) {
//             // Read list size.
//             argpos++;
//             long long ninputs;
//             if (RedisModule_StringToLongLong(argv[argpos], &ninputs) != REDISMODULE_OK) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid argument for list input count in AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             // Check validity of current list size.
//             argpos++;
//             if (argpos >= argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
//                 goto cleanup;
//             }
//             size_t first_input_pos = argpos;
//             if (first_input_pos + ninputs > argc) {
//                 RAI_SetError(error, RAI_ESCRIPTRUN,
//                              "ERR number of list input keys to AI.SCRIPTEXECUTE command does not "
//                              "match the number of given arguments");
//                 goto cleanup;
//             }
//             for (; argpos < first_input_pos + ninputs; argpos++) {
//                 inputs = array_append(inputs, RAI_HoldString(NULL, argv[argpos]));
//             }
//             RAI_ScriptRunCtxAddListSize(sctx, ninputs);
//             continue;
//         }

//         RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Unrecongnized parameter to AI.SCRIPTEXECUTE");
//         goto cleanup;
//     }
//     if (argpos != argc) {
//         RAI_SetError(error, RAI_ESCRIPTRUN, "ERR Encountered problem parsing AI.SCRIPTEXECUTE");
//         return REDISMODULE_ERR;
//     }

//     if (!_Script_buildInputsBySchema(sctx, inputs, inkeys, error)) {
//         goto cleanup;
//     }
//     array_free(inputs);

//     return REDISMODULE_OK;
// cleanup:
//     array_free(inputs);
//     return REDISMODULE_ERR;
// }
// }
// }

static int _ScriptExecuteCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                           RAI_Error *error, RedisModuleString ***inkeys,
                                           RedisModuleString ***outkeys, RAI_ScriptRunCtx *sctx,
                                           long long *timeout) {
    int argpos = 3;
    bool inputsDone = false;
    bool outputsDone = false;
    bool KeysDone = false;
    //Local input context to verify correctness.
    array_new_on_stack(RedisModuleString *, 10, inputs);

    while (argpos < argc) {
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
                inputs = array_append(inputs, RAI_HoldString(NULL, argv[argpos]));
            }
            continue;
        }
        if (!strcasecmp(arg_string, "KEYS")) {
            if(argpos != 3) {
                
            }
            // Check for already given keys.
            if (KeysDone) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Already Encountered KEYS scope in AI.SCRIPTEXECUTE command");
                goto cleanup;
            }
            KeysDone = true;
            // Read key number.
            argpos++;
            if (argpos >= argc) {
                RAI_SetError(error, RAI_ESCRIPTRUN,
                             "ERR Invalid arguments provided to AI.SCRIPTEXECUTE");
                goto cleanup;
            }
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
                if (!VerifyKeyInThisShard(ctx, argv[argpos])) {
                    RAI_SetError(error, RAI_ESCRIPTRUN,
                                 "ERR CROSSSLOT Keys in AI.SCRIPTEXECUTE request don't hash to the "
                                 "same slot");
                    goto cleanup;
                }
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

    if (!_Script_buildInputsBySchema(sctx, inputs, inkeys, error)) {
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
    RedisModuleString *scriptName = argv[1];
    RAI_GetScriptFromKeyspace(ctx, scriptName, &script, REDISMODULE_READ, error);
    if (!script) {
        goto cleanup;
    }

    RAI_DagOpSetRunKey(currentOp, RAI_HoldString(ctx, argv[1]));

    const char *func_name = ScriptCommand_GetFunction(argv[2]);
    if (!func_name) {
        RAI_SetError(rinfo->err, RAI_ESCRIPTRUN, "ERR function name not specified");
        goto cleanup;
    }

    RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(script, func_name);
    long long timeout = 0;
    size_t *listSizes = array_new(size_t, 1);
    if (_ScriptExecuteCommand_ParseArgs(ctx, argv, argc, error, &currentOp->inkeys,
                                        &currentOp->outkeys, sctx, &timeout) == REDISMODULE_ERR) {
        goto cleanup;
    }
    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(error, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }
    currentOp->sctx = sctx;
    currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    currentOp->devicestr = sctx->script->devicestr;

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ScriptRunCtx, bring inputs from key space.
        if (ScriptRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, sctx, error) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}