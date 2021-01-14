#include "dag_builder.h"
#include "run_info.h"
#include "dag_parser.h"
#include "string_utils.h"
#include "modelRun_ctx.h"

int RAI_DAGLoadTensor(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Tensor *tensor) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RedisModuleString *key_name = RedisModule_CreateString(NULL, t_name, strlen(t_name));
    // Add the tensor under its "mangled" key name to the DAG local context dict.
    char buf[16];
    sprintf(buf, "%04d", 1);
    RedisModule_StringAppendBuffer(NULL, key_name, buf, strlen(buf));
    AI_dictAdd(rinfo->dagTensorsContext, (void *)key_name,
               (void *)RAI_TensorGetShallowCopy(tensor));
    RedisModule_FreeString(NULL, key_name);

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
        if (ModelGetNumInputs(model) != array_len(op->inkeys)) {
            RAI_SetError(err, RAI_EDAGBUILDER,
                         "Number of keys given as INPUTS does not match model definition");
            return REDISMODULE_ERR;
        }
        if (ModelGetNumOutputs(model) != array_len(op->outkeys)) {
            RAI_SetError(err, RAI_EDAGBUILDER,
                         "Number of keys given as OUTPUTS does not match model definition");
            return REDISMODULE_ERR;
        }
    }
    rinfo->dagOps = array_append(rinfo->dagOps, op);

    return REDISMODULE_OK;
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

int RAI_DAGAddOpsFromString(RAI_DAGRunCtx *run_info, const char *dag, RAI_Error *err) {

    int res = REDISMODULE_ERR;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    int argc = 0;
    char dag_string[strlen(dag) + 1];
    strcpy(dag_string, dag);

    char *token = strtok(dag_string, " ");
    if (strcmp(token, "|>") != 0) {
        RAI_SetError(err, RAI_EDAGBUILDER, "DAG op should start with: '|>' ");
        return res;
    }
    RedisModuleString **argv = array_new(RedisModuleString *, 2);
    while (token != NULL) {
        RedisModuleString *RS_token = RedisModule_CreateString(NULL, token, strlen(token));
        argv = array_append(argv, RS_token);
        argc++;
        token = strtok(NULL, " ");
    }

    size_t num_ops_before = array_len(rinfo->dagOps);
    size_t new_ops = 0;
    RAI_DagOp *op;
    for (size_t i = 0; i < argc; i++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[i], NULL);
        if (strcmp(arg_string, "|>") == 0 && i < argc - 1) {
            RAI_InitDagOp(&op);
            rinfo->dagOps = array_append(rinfo->dagOps, op);
            new_ops++;
            op->argv = &argv[i + 1];
        } else {
            op->argc++;
        }
    }

    if (ParseDAGOps(rinfo, num_ops_before, new_ops) != REDISMODULE_OK) {
        // Remove all ops that where added before the error and go back to the initial state.
        RAI_SetError(err, RAI_GetErrorCode(rinfo->err), RAI_GetError(rinfo->err));
        for (size_t i = num_ops_before; i < array_len(rinfo->dagOps); i++) {
            RAI_FreeDagOp(rinfo->dagOps[i]);
        }
        rinfo->dagOps = array_trimm_len(rinfo->dagOps, num_ops_before);
        goto cleanup;
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    res = REDISMODULE_OK;

cleanup:
    for (size_t i = 0; i < argc; i++) {
        RedisModule_FreeString(NULL, argv[i]);
    }
    array_free(argv);
    return res;
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
