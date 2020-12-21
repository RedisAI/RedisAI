/**
 * script.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the PyTorch Script data structure.
 *
 */

#include "script.h"

#include "backends.h"
#include "rmutil/alloc.h"
#include "script_struct.h"
#include "stats.h"
#include "util/arr_rm_alloc.h"
#include "util/string_utils.h"
#include "version.h"
#include <pthread.h>

RAI_Script *RAI_ScriptCreate(const char *devicestr, RedisModuleString *tag, const char *scriptdef,
                             RAI_Error *err) {
    if (!RAI_backends.torch.script_create) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return NULL;
    }
    RAI_Script *script = RAI_backends.torch.script_create(devicestr, scriptdef, err);

    if (script) {
        if (tag) {
            script->tag = RAI_HoldString(NULL, tag);
        } else {
            script->tag = RedisModule_CreateString(NULL, "", 0);
        }
    }

    return script;
}

void RAI_ScriptFree(RAI_Script *script, RAI_Error *err) {
    if (__atomic_sub_fetch(&script->refCount, 1, __ATOMIC_RELAXED) > 0) {
        return;
    }

    if (!RAI_backends.torch.script_free) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return;
    }

    RedisModule_FreeString(NULL, script->tag);

    RAI_RemoveStatsEntry(script->infokey);

    RAI_backends.torch.script_free(script, err);
}

RAI_ScriptRunCtx *RAI_ScriptRunCtxCreate(RAI_Script *script, const char *fnname) {
#define PARAM_INITIAL_SIZE 10
    RAI_ScriptRunCtx *sctx = RedisModule_Calloc(1, sizeof(*sctx));
    sctx->script = RAI_ScriptGetShallowCopy(script);
    sctx->inputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
    sctx->outputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
    sctx->fnname = RedisModule_Strdup(fnname);
    sctx->variadic = -1;
    return sctx;
}

static int Script_RunCtxAddParam(RAI_ScriptRunCtx *sctx, RAI_ScriptCtxParam **paramArr,
                                 RAI_Tensor *tensor) {
    RAI_ScriptCtxParam param = {
        .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
    };
    *paramArr = array_append(*paramArr, param);
    return 1;
}

int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor, RAI_Error *err) {
    // Even if variadic is set, we still allow to add inputs in the LLAPI
    return Script_RunCtxAddParam(sctx, &sctx->inputs, inputTensor);
}

int RAI_ScriptRunCtxAddInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors, size_t len,
                                 RAI_Error *err) {
    if (sctx->variadic == -1) {
        sctx->variadic = array_len(sctx->inputs);
    } else {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED,
                     "ERR Already encountered a variable size list of tensors");
        return 0;
    }

    int res;
    for (size_t i = 0; i < len; i++) {
        res = Script_RunCtxAddParam(sctx, &sctx->inputs, inputTensors[i]);
        if (res != 1)
            return res;
    }
    return 1;
}

int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx *sctx) {
    return Script_RunCtxAddParam(sctx, &sctx->outputs, NULL);
}

size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx *sctx) { return array_len(sctx->outputs); }

RAI_Tensor *RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx *sctx, size_t index) {
    assert(RAI_ScriptRunCtxNumOutputs(sctx) > index && index >= 0);
    return sctx->outputs[index].tensor;
}

void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx *sctx, int freeTensors) {
    if (freeTensors) {
        for (size_t i = 0; i < array_len(sctx->inputs); ++i) {
            RAI_TensorFree(sctx->inputs[i].tensor);
        }

        for (size_t i = 0; i < array_len(sctx->outputs); ++i) {
            if (sctx->outputs[i].tensor) {
                RAI_TensorFree(sctx->outputs[i].tensor);
            }
        }
    }

    array_free(sctx->inputs);
    array_free(sctx->outputs);

    RedisModule_Free(sctx->fnname);

    RAI_Error err = {0};
    RAI_ScriptFree(sctx->script, &err);

    if (err.code != RAI_OK) {
        // TODO: take it to client somehow
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
    }

    RedisModule_Free(sctx);
}

int RAI_ScriptRun(RAI_ScriptRunCtx *sctx, RAI_Error *err) {
    if (!RAI_backends.torch.script_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return REDISMODULE_ERR;
    }

    return RAI_backends.torch.script_run(sctx, err);
}

RAI_Script *RAI_ScriptGetShallowCopy(RAI_Script *script) {
    __atomic_fetch_add(&script->refCount, 1, __ATOMIC_RELAXED);
    return script;
}

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Script **script, int mode) {
    *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(*key);
        RedisModule_ReplyWithError(ctx, "ERR script key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_ScriptType) {
        RedisModule_CloseKey(*key);
        RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *script = RedisModule_ModuleTypeGetValue(*key);
    return REDISMODULE_OK;
}

int RedisAI_ScriptRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                       RedisModuleString **argv, int argc) {
    RedisModule_KeyAtPos(ctx, 1);
    size_t startpos = 3;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    const char *str = RedisModule_StringPtrLen(argv[startpos], NULL);
    if (!strcasecmp(str, "TIMEOUT")) {
        startpos += 2;
    }
    startpos += 1;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    for (size_t argpos = startpos; argpos < argc; argpos++) {
        str = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(str, "OUTPUTS")) {
            continue;
        }
        RedisModule_KeyAtPos(ctx, argpos);
    }
    return REDISMODULE_OK;
}

/**
 * AI.SCRIPTRUN <key> <function> INPUTS <input> [input ...] OUTPUTS <output>
 * [output ...]
 */
int RedisAI_Parse_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                         RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                         int *variadic, RAI_Error *error) {
    if (argc < 6) {
        RAI_SetError(error, RAI_ESCRIPTRUN,
                     "ERR wrong number of arguments for 'AI.SCRIPTRUN' command");
        return -1;
    }

    const char *inputstr = RedisModule_StringPtrLen(argv[2], NULL);
    if (strcasecmp(inputstr, "INPUTS") == 0) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR function name not specified");
        return -1;
    }

    inputstr = RedisModule_StringPtrLen(argv[3], NULL);
    if (strcasecmp(inputstr, "INPUTS")) {
        RAI_SetError(error, RAI_ESCRIPTRUN, "ERR INPUTS not specified");
        return -1;
    }

    // parsing aux vars
    int is_output = 0;
    int outputs_flag_count = 0;
    int variadic_ = *variadic;
    size_t argpos = 4;
    for (; argpos <= argc - 1; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!arg_string) {
            RAI_SetError(error, RAI_ESCRIPTRUN, "ERR NULL argument on SCRIPTRUN");
            return -1;
        }
        if (!strcasecmp(arg_string, "OUTPUTS") && outputs_flag_count == 0) {
            is_output = 1;
            outputs_flag_count = 1;
        } else {
            if (!strcasecmp(arg_string, "$")) {
                if (variadic_ > -1) {
                    RAI_SetError(error, RAI_ESCRIPTRUN,
                                 "ERR Already encountered a variable size list of tensors");
                    return -1;
                }
                variadic_ = argpos - 4;
                continue;
            }
            RedisModule_RetainString(ctx, argv[argpos]);
            if (is_output == 0) {
                *inkeys = array_append(*inkeys, argv[argpos]);
            } else {
                *outkeys = array_append(*outkeys, argv[argpos]);
            }
        }
    }

    *variadic = variadic_;

    return argpos;
}

RedisModuleType *RAI_ScriptRedisType(void) { return RedisAI_ScriptType; }
