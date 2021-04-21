/**
 * script.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the PyTorch Script data structure.
 *
 */

#include <pthread.h>
#include "version.h"
#include "script.h"
#include "script_struct.h"
#include "stats.h"
#include "util/arr.h"
#include "util/string_utils.h"
#include "rmutil/alloc.h"
#include "backends/backends.h"
#include "execution/DAG/dag.h"
#include "execution/run_info.h"

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
    sctx->listSizes = array_new(size_t, PARAM_INITIAL_SIZE);
    sctx->keys = array_new(RedisModuleString*, PARAM_INITIAL_SIZE);
    return sctx;
}

static int _Script_RunCtxAddParam(RAI_ScriptCtxParam **paramArr, RAI_Tensor *tensor) {
    RAI_ScriptCtxParam param = {
        .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
    };
    *paramArr = array_append(*paramArr, param);
    return 1;
}

int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor, RAI_Error *error) {
    // Even if variadic is set, we still allow to add inputs in the LLAPI
    _Script_RunCtxAddParam(&sctx->inputs, inputTensor);
    return 1;
}

int RAI_ScriptRunCtxAddInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors, size_t len,
                                 RAI_Error *err) {
    int res;
    for (size_t i = 0; i < len; i++) {
        res = _Script_RunCtxAddParam(&sctx->inputs, inputTensors[i]);
    }
    sctx->listSizes = array_append(sctx->listSizes, len);
    return 1;
}

int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx *sctx) {
    return _Script_RunCtxAddParam(&sctx->outputs, NULL);
}

size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx *sctx) { return array_len(sctx->outputs); }

RAI_Tensor *RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx *sctx, size_t index) {
    assert(RAI_ScriptRunCtxNumOutputs(sctx) > index && index >= 0);
    return sctx->outputs[index].tensor;
}

void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx *sctx) {

    for (size_t i = 0; i < array_len(sctx->inputs); ++i) {
        RAI_TensorFree(sctx->inputs[i].tensor);
    }

    for (size_t i = 0; i < array_len(sctx->outputs); ++i) {
        if (sctx->outputs[i].tensor) {
            RAI_TensorFree(sctx->outputs[i].tensor);
        }
    }

    RedisModuleCtx* ctx = RedisModule_ThreadSafeContext(NULL);
    for (size_t i = 0; i < array_len(sctx->keys); ++i) {
        RedisModule_FreeString(ctx, sctx->keys[i]);
    }
    RedisModule_FreeThreadSafeContext(ctx);

    array_free(sctx->inputs);
    array_free(sctx->outputs);
    array_free(sctx->listSizes);
    array_free(sctx->keys);


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
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Script **script,
                              int mode, RAI_Error *err) {
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(key);
        RAI_SetError(err, RAI_EKEYEMPTY, "ERR script key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(key) != RedisAI_ScriptType) {
        RedisModule_CloseKey(key);
        RAI_SetError(err, RAI_ESCRIPTRUN, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *script = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
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

int RedisAI_ScriptExecute_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                           RedisModuleString **argv, int argc) {
    RedisModule_KeyAtPos(ctx, 1);
    size_t argpos = 3;
    if (argpos >= argc) {
        return REDISMODULE_ERR;
    }
    long long count;
    while (argpos < argc) {
        const char *str = RedisModule_StringPtrLen(argv[argpos++], NULL);

        // Inputs, outpus, keys, lists.
        if ((!strcasecmp(str, "INPUTS")) || (!strcasecmp(str, "OUTPUTS")) ||
            (!strcasecmp(str, "LIST_INPUT")) || (!strcasecmp(str, "KEYS"))) {
            if (argpos >= argc) {
                return REDISMODULE_ERR;
            }
            if (RedisModule_StringToLongLong(argv[argpos++], &count) != REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            if (count < 0) {
                return REDISMODULE_ERR;
            }
            if (argpos + count >= argc) {
                return REDISMODULE_ERR;
            }
            for (long long i = 0; i < count; i++) {
                RedisModule_KeyAtPos(ctx, argpos++);
            }
            continue;
        }
        // Timeout
        if (!strcasecmp(str, "TIMEOUT")) {
            argpos++;
            break;
        }
        // Undefinded input.
        return REDISMODULE_ERR;
    }
    if (argpos != argc) {
        return REDISMODULE_ERR;
    } else {
        return REDISMODULE_OK;
    }
}

RedisModuleType *RAI_ScriptRedisType(void) { return RedisAI_ScriptType; }

int RAI_ScriptRunAsync(RAI_ScriptRunCtx *sctx, RAI_OnFinishCB ScriptAsyncFinish,
                       void *private_data) {

    RedisAI_RunInfo *rinfo = NULL;
    RAI_InitRunInfo(&rinfo);

    rinfo->single_op_dag = 1;
    rinfo->OnFinish = (RedisAI_OnFinishCB)ScriptAsyncFinish;
    rinfo->private_data = private_data;

    RAI_DagOp *op;
    RAI_InitDagOp(&op);

    op->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    op->devicestr = sctx->script->devicestr;
    op->sctx = sctx;

    rinfo->dagOps = array_append(rinfo->dagOps, op);
    rinfo->dagOpCount = 1;
    if (DAG_InsertDAGToQueue(rinfo) != REDISMODULE_OK) {
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}
