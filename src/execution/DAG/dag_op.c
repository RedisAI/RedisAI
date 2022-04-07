#include "dag_op.h"
#include "util/arr.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/execution_contexts/scriptRun_ctx.h"
/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitDagOp(RAI_DagOp **result) {
    RAI_DagOp *dagOp;
    dagOp = (RAI_DagOp *)RedisModule_Calloc(1, sizeof(RAI_DagOp));

    dagOp->commandType = REDISAI_DAG_CMD_NONE;
    dagOp->inkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
    dagOp->outkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
    dagOp->inkeys_indices = array_new(size_t, 1);
    dagOp->outkeys_indices = array_new(size_t, 1);
    dagOp->outTensor = NULL;
    dagOp->ectx = NULL;
    dagOp->devicestr = NULL;
    dagOp->duration_us = 0;
    dagOp->result = -1;
    RAI_InitError(&dagOp->err);
    dagOp->argv = NULL;
    dagOp->argc = 0;

    *result = dagOp;
    return REDISMODULE_OK;
}

void RAI_FreeDagOp(RAI_DagOp *dagOp) {

    RAI_FreeError(dagOp->err);

    if (dagOp->outTensor)
        RAI_TensorFree(dagOp->outTensor);

    if (dagOp->ectx) {
        dagOp->ectx->freeFn(dagOp->ectx);
    }

    if (dagOp->inkeys) {
        for (size_t i = 0; i < array_len(dagOp->inkeys); i++) {
            RedisModule_FreeString(NULL, dagOp->inkeys[i]);
        }
        array_free(dagOp->inkeys);
    }
    array_free(dagOp->inkeys_indices);

    if (dagOp->outkeys) {
        for (size_t i = 0; i < array_len(dagOp->outkeys); i++) {
            RedisModule_FreeString(NULL, dagOp->outkeys[i]);
        }
        array_free(dagOp->outkeys);
    }
    array_free(dagOp->outkeys_indices);
    RedisModule_Free(dagOp);
}
