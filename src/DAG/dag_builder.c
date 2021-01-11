#include "dag_builder.h"
#include "run_info.h"
#include "string_utils.h"
#include "modelRun_ctx.h"

static int _LoadTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                                   RedisModuleKey **key, RAI_Tensor **tensor, RAI_Error *err) {

    int res = REDISMODULE_ERR;
    *key = RedisModule_OpenKey(ctx, keyName, REDISMODULE_READ);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RAI_SetError(err, RAI_EDAGBUILDER, "ERR tensor key is empty");
        goto end;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RAI_SetError(err, RAI_EDAGBUILDER, REDISMODULE_ERRORMSG_WRONGTYPE);
        goto end;
    }
    *tensor = RedisModule_ModuleTypeGetValue(*key);
    res = REDISMODULE_OK;

end:
    RedisModule_CloseKey(*key);
    return res;
}

static int _RAI_DagLoadTensor(RAI_DAGRunCtx *run_info, RedisModuleString *key_name,
                              RAI_Error *err) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RAI_Tensor *t;
    RedisModuleKey *key;
    if (_LoadTensorFromKeyspace(ctx, key_name, &key, &t, err) == REDISMODULE_ERR) {
        RedisModule_FreeString(NULL, key_name);
        RedisModule_FreeThreadSafeContext(ctx);
        return REDISMODULE_ERR;
    }
    // Add the tensor under its "mangled" key name to the DAG local context dict.
    char buf[16];
    sprintf(buf, "%04d", 1);
    RedisModule_StringAppendBuffer(NULL, key_name, buf, strlen(buf));
    AI_dictAdd(rinfo->dagTensorsContext, (void *)key_name, (void *)RAI_TensorGetShallowCopy(t));
    RedisModule_FreeString(NULL, key_name);
    RedisModule_FreeThreadSafeContext(ctx);
    return REDISMODULE_OK;
}

RAI_DAGRunCtx *RAI_DAGRunCtxCreate(void) {
    RedisAI_RunInfo *rinfo;
    RAI_InitRunInfo(&rinfo);
    return (RAI_DAGRunCtx *)rinfo;
}

RAI_DAGRunOp *RAI_DAGCreateModelRunOp(RAI_Model *model) {
    RAI_ModelRunCtx *mctx = RAI_ModelRunCtxCreate(model);
    RAI_DagOp *op;
    RAI_InitDagOp(&op);

    op->commandType = REDISAI_DAG_CMD_MODELRUN;
    op->mctx = mctx;
    op->devicestr = model->devicestr;
    op->runkey = RAI_HoldString(NULL, (RedisModuleString *)model->infokey);
    return (RAI_DAGRunOp *)op;
}

RAI_DAGRunOp *RAI_DAGCreateScriptRunOp(RAI_Script *script, const char *func_name) {
    RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(script, func_name);
    RAI_DagOp *op;
    RAI_InitDagOp(&op);

    op->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
    op->sctx = sctx;
    op->devicestr = script->devicestr;
    op->runkey = RAI_HoldString(NULL, (RedisModuleString *)script->infokey);
    return (RAI_DAGRunOp *)op;
}

int RAI_DAGRunOpAddInput(RAI_DAGRunOp *DAGOp, const char *input) {
    RAI_DagOp *op = (RAI_DagOp *)DAGOp;
    RedisModuleString *inkey = RedisModule_CreateString(NULL, input, strlen(input));
    op->inkeys = array_append(op->inkeys, inkey);
    return REDISMODULE_OK;
}

int RAI_DAGRunOpAddOutput(RAI_DAGRunOp *DAGOp, const char *output) {
    RAI_DagOp *op = (RAI_DagOp *)DAGOp;
    RedisModuleString *outkey = RedisModule_CreateString(NULL, output, strlen(output));
    op->outkeys = array_append(op->outkeys, outkey);
    return REDISMODULE_OK;
}

int RAI_DAGAddRunOp(RAI_DAGRunCtx *run_info, RAI_DAGRunOp *DAGop, RAI_Error *err) {

    RAI_DagOp *op = (RAI_DagOp *)DAGop;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    if (op->mctx) {
        RAI_Model *model = op->mctx->model;
        if (model->ninputs != array_len(op->inkeys)) {
            RAI_SetError(err, RAI_EDAGBUILDER,
                         "Number of keys given as INPUTS does not match model definition");
            return REDISMODULE_ERR;
        }
        if (model->noutputs != array_len(op->outkeys)) {
            RAI_SetError(err, RAI_EDAGBUILDER,
                         "Number of keys given as OUTPUTS does not match model definition");
            return REDISMODULE_ERR;
        }
    }
    rinfo->dagOps = array_append(rinfo->dagOps, op);

    return REDISMODULE_OK;
}

int RAI_DAGLoadTensor(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err) {

    RedisModuleString *key_name = RedisModule_CreateString(NULL, t_name, strlen(t_name));
    return _RAI_DagLoadTensor(run_info, key_name, err);
}

int RAI_DAGLoadTensorRS(RAI_DAGRunCtx *run_info, RedisModuleString *t_name, RAI_Error *err) {

    RedisModuleString *key_name = RedisModule_CreateStringFromString(NULL, t_name);
    return _RAI_DagLoadTensor(run_info, key_name, err);
}

int RAI_DAGAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err) {

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

int RAI_DAGAddTensorSet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Tensor *tensor) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RAI_DagOp *op;
    RAI_InitDagOp(&op);
    rinfo->dagOps = array_append(rinfo->dagOps, op);
    op->commandType = REDISAI_DAG_CMD_TENSORSET;
    op->devicestr = "CPU";
    RedisModuleString *name = RedisModule_CreateString(NULL, t_name, strlen(t_name));
    op->outkeys = array_append(op->outkeys, name);
    op->outTensor = RAI_TensorGetShallowCopy(tensor);
    return REDISMODULE_OK;
}

size_t RAI_DAGNumOps(RAI_DAGRunCtx *run_info) {
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    return array_len(rinfo->dagOps);
}

void RAI_DAGRunOpFree(RAI_DAGRunOp *dagOp) {
    RAI_DagOp *op = (RAI_DagOp *)dagOp;
    RAI_FreeDagOp(op);
}

void RAI_DAGFree(RAI_DAGRunCtx *run_info) {
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RAI_FreeRunInfo(rinfo);
}
