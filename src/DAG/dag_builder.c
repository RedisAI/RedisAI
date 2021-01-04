#include "dag_builder.h"
#include "run_info.h"
#include "string_utils.h"

int _LoadTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                            RAI_Tensor **tensor, RAI_Error *err) {

    *key = RedisModule_OpenKey(ctx, keyName, REDISMODULE_READ);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(*key);
        RAI_SetError(err, RAI_EDAGBUILDER, "ERR tensor key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RedisModule_CloseKey(*key);
        RAI_SetError(err, RAI_EDAGBUILDER, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *tensor = RedisModule_ModuleTypeGetValue(*key);
    RedisModule_CloseKey(*key);
    return REDISMODULE_OK;
}

RAI_DAGRunCtx *RAI_DagRunCtxCreate(void) {
    RedisAI_RunInfo *rinfo;
    RAI_InitRunInfo(&rinfo);
    return (RAI_DAGRunCtx *)rinfo;
}

int RAI_DagAddModelRun_(RAI_DAGRunCtx *run_info, RAI_ModelRunCtx *mctx, RedisModuleString **inputs,
                        RedisModuleString **outputs, RAI_Error *err) {
    if (array_len(mctx->inputs) != 0 || array_len(mctx->outputs) != 0) {
        RAI_SetError(
            err, RAI_EDAGBUILDER,
            "Model run context cannot contain inputs or outputs when it is a part of a DAG");
        return REDISMODULE_ERR;
    }
    RAI_Model *model = mctx->model;
    if (model->ninputs != array_len(inputs)) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "Number of keys given as INPUTS does not match model definition");
        return REDISMODULE_ERR;
    }
    if (model->noutputs != array_len(outputs)) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "Number of keys given as OUTPUTS does not match model definition");
        return REDISMODULE_ERR;
    }

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RAI_DagOp *op;
    RAI_InitDagOp(&op);
    rinfo->dagOps = array_append(rinfo->dagOps, op);

    op->commandType = REDISAI_DAG_CMD_MODELRUN;
    op->mctx = mctx;
    op->devicestr = model->devicestr;
    op->inkeys = inputs;
    op->outkeys = outputs;
    op->runkey = RAI_HoldString(NULL, (RedisModuleString *)model->infokey);
    return REDISMODULE_OK;
}

int RAI_DagAddModelRun(RAI_DAGRunCtx *run_info, RAI_ModelRunCtx *mctx, const char **inputs,
                       size_t ninputs, const char **outputs, size_t noutputs, RAI_Error *err) {

    RedisModuleString **inkeys = array_new(RedisModuleString *, 1);
    for (size_t i = 0; i < ninputs; i++) {
        RedisModuleString *inkey = RedisModule_CreateString(NULL, inputs[i], strlen(inputs[i]));
        inkeys = array_append(inkeys, inkey);
    }
    RedisModuleString **outkeys = array_new(RedisModuleString *, 1);
    for (size_t i = 0; i < noutputs; i++) {
        RedisModuleString *outkey = RedisModule_CreateString(NULL, outputs[i], strlen(outputs[i]));
        outkeys = array_append(outkeys, outkey);
    }
    return RAI_DagAddModelRun_(run_info, mctx, inkeys, outkeys, err);
}

int RedisAI_DagAddLoadPhase_(RAI_DAGRunCtx *run_info, RedisModuleString **keys_to_load,
                             RAI_Error *err) {

    int status = REDISMODULE_ERR;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RedisModule_ThreadSafeContextLock(ctx);
    size_t n_keys = array_len(keys_to_load);

    for (size_t i = 0; i < n_keys; i++) {
        RAI_Tensor *t;
        RedisModuleKey *key;
        if (_LoadTensorFromKeyspace(ctx, keys_to_load[i], &key, &t, err) == REDISMODULE_ERR) {
            goto cleanup;
        }
        // Add the tensor under its "mangled" key name to the DAG local context dict.
        char buf[16];
        sprintf(buf, "%04d", 1);
        RedisModule_StringAppendBuffer(NULL, keys_to_load[i], buf, strlen(buf));
        AI_dictAdd(rinfo->dagTensorsContext, (void *)keys_to_load[i],
                   (void *)RAI_TensorGetShallowCopy(t));
    }
    status = REDISMODULE_OK;

cleanup:
    RedisModule_ThreadSafeContextUnlock(ctx);
    for (size_t i = 0; i < n_keys; i++) {
        RedisModule_FreeString(NULL, keys_to_load[i]);
    }
    array_free(keys_to_load);
    return status;
}

int RedisAI_DagAddLoadPhase(RAI_DAGRunCtx *run_info, const char **t_names, uint n, RAI_Error *err) {
    if (n == 0) {
        RAI_SetError(err, RAI_EDAGBUILDER, "Number of keys to LOAD must be positive");
        return REDISMODULE_ERR;
    }
    RedisModuleString **keys_to_load = array_new(RedisModuleString *, 1);
    for (size_t i = 0; i < n; i++) {
        RedisModuleString *key = RedisModule_CreateString(NULL, t_names[i], strlen(t_names[i]));
        keys_to_load = array_append(keys_to_load, key);
    }
    return RedisAI_DagAddLoadPhase_(run_info, keys_to_load, err);
}

int RAI_DagAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RAI_DagOp *op;
    RAI_InitDagOp(&op);
    rinfo->dagOps = array_append(rinfo->dagOps, op);
    op->commandType = REDISAI_DAG_CMD_TENSORGET;
    op->devicestr = "CPU";
    RedisModuleString *name = RedisModule_CreateString(NULL, t_name, strlen(t_name));
    op->inkeys = array_append(op->inkeys, name);
    return REDISMODULE_OK;
}