#include "dag_builder.h"
#include "execution/parsing/dag_parser.h"
#include "util/string_utils.h"
#include "execution/run_info.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/execution_contexts/scriptRun_ctx.h"

// Store the given arguments from the string in argv array and their amount in argc.
int _StringToRMArray(const char *dag, RedisModuleString ***argv, int *argc, RAI_Error *err) {

    char dag_string[strlen(dag) + 1];
    strcpy(dag_string, dag);

    char *token = strtok(dag_string, " ");
    if (strcmp(token, "|>") != 0) {
        RAI_SetError(err, RAI_EDAGBUILDER, "DAG op should start with: '|>' ");
        return REDISMODULE_ERR;
    }

    while (token != NULL) {
        RedisModuleString *RS_token = RedisModule_CreateString(NULL, token, strlen(token));
        *argv = array_append(*argv, RS_token);
        (*argc)++;
        token = strtok(NULL, " ");
    }
    return REDISMODULE_OK;
}

int RAI_DAGLoadTensor(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Tensor *tensor) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    RedisModuleString *key_name = RedisModule_CreateString(NULL, t_name, strlen(t_name));

    // Cannot load more than one tensor under the same name
    if (AI_dictFind(rinfo->tensorsNamesToIndices, key_name) != NULL) {
        RedisModule_FreeString(NULL, key_name);
        return REDISMODULE_ERR;
    }

    // Add the tensor to the DAG shared tensors and map its name to the relevant index.
    size_t index = array_len(rinfo->dagSharedTensors);
    AI_dictAdd(rinfo->tensorsNamesToIndices, (void *)key_name, (void *)index);
    RAI_TensorGetShallowCopy(tensor);
    rinfo->dagSharedTensors = array_append(rinfo->dagSharedTensors, (void *)tensor);
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

int RAI_DAGAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name) {

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
    array_new_on_stack(RAI_DagOp *, 10, new_ops);
    array_new_on_stack(RedisModuleString *, 100, argv);
    int argc = 0;
    if (_StringToRMArray(dag, &argv, &argc, err) != REDISMODULE_OK) {
        goto cleanup;
    }

    RAI_DagOp *op;
    for (size_t i = 0; i < argc; i++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[i], NULL);
        if (strcmp(arg_string, "|>") == 0 && i < argc - 1) {
            RAI_InitDagOp(&op);
            new_ops = array_append(new_ops, op);
            op->argv = &argv[i + 1];
        } else {
            op->argc++;
        }
    }

    if (ParseDAGOps(rinfo, new_ops) != REDISMODULE_OK) {
        RAI_SetError(err, RAI_GetErrorCode(rinfo->err), RAI_GetError(rinfo->err));
        goto cleanup;
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    res = REDISMODULE_OK;

cleanup:
    array_free(new_ops);
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
