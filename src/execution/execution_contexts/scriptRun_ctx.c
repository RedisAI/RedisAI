#include "scriptRun_ctx.h"
#include "redismodule.h"
#include "execution/utils.h"
#include "execution/DAG/dag.h"
#include "execution/run_info.h"
#include "backends/backends.h"
#include "util/string_utils.h"

RAI_ScriptRunCtx *RAI_ScriptRunCtxCreate(RAI_Script *script, const char *fnname) {
#define PARAM_INITIAL_SIZE 10
    RAI_ScriptRunCtx *sctx = RedisModule_Calloc(1, sizeof(*sctx));
    sctx->script = RAI_ScriptGetShallowCopy(script);
    sctx->inputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
    sctx->outputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
    sctx->fnname = RedisModule_Strdup(fnname);
    sctx->listSizes = array_new(size_t, PARAM_INITIAL_SIZE);
    sctx->intInputs = array_new(int32_t, PARAM_INITIAL_SIZE);
    sctx->floatInputs = array_new(float, PARAM_INITIAL_SIZE);
    sctx->stringInputs = array_new(RedisModuleString *, PARAM_INITIAL_SIZE);
    return sctx;
}

static int _Script_RunCtxAddParam(RAI_ScriptCtxParam **paramArr, RAI_Tensor *tensor) {
    RAI_ScriptCtxParam param = {
        .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
    };
    *paramArr = array_append(*paramArr, param);
    return 1;
}

// Deprecated.
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor, RAI_Error *error) {
    // Even if variadic is set, we still allow to add inputs in the LLAPI
    _Script_RunCtxAddParam(&sctx->inputs, inputTensor);
    return 1;
}

// Deprecated.
int RAI_ScriptRunCtxAddInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors, size_t len,
                                 RAI_Error *err) {
    int res;
    for (size_t i = 0; i < len; i++) {
        res = _Script_RunCtxAddParam(&sctx->inputs, inputTensors[i]);
    }
    sctx->listSizes = array_append(sctx->listSizes, len);
    return 1;
}

int RAI_ScriptRunCtxAddTensorInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor) {
    _Script_RunCtxAddParam(&sctx->inputs, inputTensor);
    return 1;
}

int RAI_ScriptRunCtxAddIntInput(RAI_ScriptRunCtx *sctx, int32_t i) {
    sctx->intInputs = array_append(sctx->intInputs, i);
    return 1;
}

int RAI_ScriptRunCtxAddFloatInput(RAI_ScriptRunCtx *sctx, float f) {
    sctx->floatInputs = array_append(sctx->floatInputs, f);
    return 1;
}

int RAI_ScriptRunCtxAddRStringInput(RAI_ScriptRunCtx *sctx, RedisModuleString *s) {
    sctx->stringInputs = array_append(sctx->stringInputs, RAI_HoldString(s));
    return 1;
}

int RAI_ScriptRunCtxAddStringInput(RAI_ScriptRunCtx *sctx, const char *s, size_t len) {
    RedisModuleString *rs = RedisModule_CreateString(NULL, s, len);
    return RAI_ScriptRunCtxAddRStringInput(sctx, rs);
}

int RAI_ScriptRunCtxAddTensorInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors,
                                       size_t count) {
    int res = 1;
    for (size_t i = 0; i < count; i++) {
        res &= RAI_ScriptRunCtxAddTensorInput(sctx, inputTensors[i]);
    }
    return res;
}

int RAI_ScriptRunCtxAddIntInputList(RAI_ScriptRunCtx *sctx, int32_t *intInputs, size_t count) {
    int res = 1;
    for (size_t i = 0; i < count; i++) {
        res &= RAI_ScriptRunCtxAddIntInput(sctx, intInputs[i]);
    }
    return res;
}

int RAI_ScriptRunCtxAddFloatInputList(RAI_ScriptRunCtx *sctx, float *floatInputs, size_t count) {
    int res = 1;
    for (size_t i = 0; i < count; i++) {
        res &= RAI_ScriptRunCtxAddFloatInput(sctx, floatInputs[i]);
    }
    return res;
}

int RAI_ScriptRunCtxAddRStringInputList(RAI_ScriptRunCtx *sctx, RedisModuleString **stringInputs,
                                        size_t count) {
    int res = 1;
    for (size_t i = 0; i < count; i++) {
        res &= RAI_ScriptRunCtxAddRStringInput(sctx, stringInputs[i]);
    }
    return res;
}

int RAI_ScriptRunCtxAddStringInputList(RAI_ScriptRunCtx *sctx, const char **stringInputs,
                                       size_t *lens, size_t count) {
    int res = 1;
    for (size_t i = 0; i < count; i++) {
        res &= RAI_ScriptRunCtxAddStringInput(sctx, stringInputs[i], lens[i]);
    }
    return res;
}

int RAI_ScriptRunCtxAddListSize(RAI_ScriptRunCtx *sctx, size_t len) {
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

    for (size_t i = 0; i < array_len(sctx->stringInputs); ++i) {
        RedisModule_FreeString(NULL, sctx->stringInputs[i]);
    }

    array_free(sctx->inputs);
    array_free(sctx->outputs);
    array_free(sctx->listSizes);
    array_free(sctx->stringInputs);
    array_free(sctx->intInputs);
    array_free(sctx->floatInputs);

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

TorchScriptFunctionArgumentType *RAI_ScriptRunCtxGetSignature(RAI_ScriptRunCtx *sctx) {
    return AI_dictFetchValue(sctx->script->functionData, sctx->fnname);
}

size_t RAI_ScriptRunCtxGetInputListLen(RAI_ScriptRunCtx *sctx, size_t index) {
    return sctx->listSizes[index];
}

int ScriptRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                           RedisModuleString **outkeys, RAI_ScriptRunCtx *sctx, RAI_Error *err) {

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
