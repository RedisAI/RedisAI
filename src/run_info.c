/**
 * run_info.c
 *
 * Contains the methods to create, initialize, get, reset, and
 * free the structures that represent the context in which RedisAI blocking
 * commands operate, namely RedisAI_RunInfo and the newly added RAI_DagOp.
 *
 */

#include "err.h"
#include "model.h"
#include "modelRun_ctx.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include <pthread.h>

static void RAI_TensorDictValFree(void *privdata, void *obj) {
    return RAI_TensorFree((RAI_Tensor *)obj);
}

AI_dictType AI_dictTypeTensorVals = {
    .hashFunction = RAI_RStringsHashFunction,
    .keyDup = RAI_RStringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_RStringsKeyCompare,
    .keyDestructor = RAI_RStringsKeyDestructor,
    .valDestructor = RAI_TensorDictValFree,
};

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
    dagOp->runkey = NULL;
    dagOp->inkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
    dagOp->outkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
    dagOp->outTensors = (RAI_Tensor **)array_new(RAI_Tensor *, 1);
    dagOp->mctx = NULL;
    dagOp->sctx = NULL;
    dagOp->devicestr = NULL;
    dagOp->duration_us = 0;
    dagOp->result = -1;
    RAI_InitError(&dagOp->err);
    dagOp->argv = (RedisModuleString **)array_new(RedisModuleString *, 1);
    dagOp->argc = 0;

    *result = dagOp;
    return REDISMODULE_OK;
}

/**
 * Allocate the memory and initialise the RedisAI_RunInfo.
 * @param result Output parameter to capture allocated RedisAI_RunInfo.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitRunInfo(RedisAI_RunInfo **result) {
    RedisAI_RunInfo *rinfo;
    rinfo = (RedisAI_RunInfo *)RedisModule_Calloc(1, sizeof(RedisAI_RunInfo));

    rinfo->dagTensorsContext = AI_dictCreate(&AI_dictTypeTensorVals, NULL);
    rinfo->dagTensorsLoadedContext = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
    rinfo->dagTensorsPersistedContext = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);

    rinfo->dagOps = (RAI_DagOp **)array_new(RAI_DagOp *, 1);
    rinfo->dagError = RedisModule_Calloc(1, sizeof(int));
    RAI_InitError(&rinfo->err);
    rinfo->dagLock = RedisModule_Alloc(sizeof(pthread_rwlock_t));
    rinfo->dagRefCount = RedisModule_Calloc(1, sizeof(long long));
    rinfo->dagOpCount = 0;
    rinfo->dagCompleteOpCount = RedisModule_Calloc(1, sizeof(long long));
    rinfo->dagDeviceOpCount = 0;
    rinfo->dagDeviceCompleteOpCount = 0;
    rinfo->orig_copy = rinfo;
    pthread_rwlock_init(rinfo->dagLock, NULL);
    rinfo->timedOut = RedisModule_Calloc(1, sizeof(int));

    *result = rinfo;
    return REDISMODULE_OK;
}

int RAI_ShallowCopyDagRunInfo(RedisAI_RunInfo **result, RedisAI_RunInfo *src) {
    RedisAI_RunInfo *rinfo;
    rinfo = (RedisAI_RunInfo *)RedisModule_Alloc(sizeof(RedisAI_RunInfo));
    memcpy(rinfo, src, sizeof(RedisAI_RunInfo));

    rinfo->dagDeviceOps = (RAI_DagOp **)array_new(RAI_DagOp *, 1);
    (*rinfo->dagRefCount)++;
    rinfo->dagDeviceOpCount = 0;
    rinfo->dagDeviceCompleteOpCount = 0;
    *result = rinfo;
    return REDISMODULE_OK;
}

void RAI_FreeDagOp(RAI_DagOp *dagOp) {
    if (dagOp) {
        RAI_FreeError(dagOp->err);
        if (dagOp->argv) {
            for (size_t i = 0; i < array_len(dagOp->argv); i++) {
                RedisModule_FreeString(NULL, dagOp->argv[i]);
            }
            array_free(dagOp->argv);
        }
        // dagOp->inkeys is released on all argv release above
        // dagOp->outkeys is released on all argv release above
        // dagOp->outTensors is released on RunInfo after checking what tensors to
        // persist
        for (size_t i = 0; i < array_len(dagOp->outTensors); i++) {
            RAI_TensorFree(dagOp->outTensors[i]);
        }
        array_free(dagOp->outTensors);

        if (dagOp->mctx) {
            RAI_ModelRunCtxFree(dagOp->mctx);
        }
        if (dagOp->sctx) {
            RAI_ScriptRunCtxFree(dagOp->sctx);
        }

        if (dagOp->inkeys) {
            for (size_t i = 0; i < array_len(dagOp->inkeys); i++) {
                RedisModule_FreeString(NULL, dagOp->inkeys[i]);
            }
            array_free(dagOp->inkeys);
        }

        if (dagOp->outkeys) {
            for (size_t i = 0; i < array_len(dagOp->outkeys); i++) {
                RedisModule_FreeString(NULL, dagOp->outkeys[i]);
            }
            array_free(dagOp->outkeys);
        }
        RedisModule_Free(dagOp);
    }
}

long long RAI_DagRunInfoFreeShallowCopy(RedisAI_RunInfo *rinfo) {
    long long ref_count = __atomic_sub_fetch(rinfo->dagRefCount, 1, __ATOMIC_RELAXED);
    RedisModule_Assert(ref_count >= 0 && "Tried to free the original RunInfo object");
    if (rinfo->dagDeviceOps) {
        array_free(rinfo->dagDeviceOps);
    }
    RedisModule_Free(rinfo);
    return ref_count;
}

void RAI_FreeRunInfo(struct RedisAI_RunInfo *rinfo) {
    if (!rinfo) {
        return;
    }
    long long ref_count = *rinfo->dagRefCount;
    RedisModule_Assert(ref_count == 0);
    pthread_rwlock_destroy(rinfo->dagLock);
    RedisModule_Free(rinfo->dagLock);

    if (rinfo->dagTensorsContext) {
        AI_dictRelease(rinfo->dagTensorsContext);
        AI_dictRelease(rinfo->dagTensorsLoadedContext);
        AI_dictRelease(rinfo->dagTensorsPersistedContext);
    }

    if (rinfo->dagOps) {
        for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
            RAI_FreeDagOp(rinfo->dagOps[i]);
        }
        array_free(rinfo->dagOps);
    }

    if (rinfo->dagError) {
        RedisModule_Free(rinfo->dagError);
    }
    RAI_FreeError(rinfo->err);
    RedisModule_Free(rinfo->dagRefCount);
    RedisModule_Free(rinfo->dagCompleteOpCount);
    RedisModule_Free(rinfo->timedOut);

    RedisModule_Free(rinfo);
}

void RAI_ContextReadLock(RedisAI_RunInfo *rinfo) {
    if (rinfo->single_op_dag || rinfo->single_device_dag) {
        return;
    }
    pthread_rwlock_rdlock(rinfo->dagLock);
}

void RAI_ContextWriteLock(RedisAI_RunInfo *rinfo) {
    if (rinfo->single_op_dag || rinfo->single_device_dag) {
        return;
    }
    pthread_rwlock_wrlock(rinfo->dagLock);
}

void RAI_ContextUnlock(RedisAI_RunInfo *rinfo) {
    if (rinfo->single_op_dag || rinfo->single_device_dag) {
        return;
    }
    pthread_rwlock_unlock(rinfo->dagLock);
}

size_t RAI_RunInfoBatchSize(struct RAI_DagOp *op) {
    if (op->mctx == NULL) {
        return -1;
    }

    size_t ninputs = RAI_ModelRunCtxNumInputs(op->mctx);

    int batchsize = 0;

    if (ninputs == 0) {
        return batchsize;
    }

    for (size_t i = 0; i < ninputs; i++) {
        RAI_Tensor *input = RAI_ModelRunCtxInputTensor(op->mctx, i);

        if (i == 0) {
            batchsize = RAI_TensorDim(input, 0);
            continue;
        }

        if (batchsize != RAI_TensorDim(input, 0)) {
            batchsize = 0;
            break;
        }
    }

    return batchsize;
}

int RAI_RunInfoBatchable(struct RAI_DagOp *op1, struct RAI_DagOp *op2) {

    if (op1->mctx == NULL || op2->mctx == NULL) {
        return 0;
    }

    if (op1->mctx->model != op2->mctx->model) {
        return 0;
    }

    const int ninputs1 = RAI_ModelRunCtxNumInputs(op1->mctx);
    const int ninputs2 = RAI_ModelRunCtxNumInputs(op2->mctx);

    if (ninputs1 != ninputs2) {
        return 0;
    }

    for (int i = 0; i < ninputs1; i++) {
        RAI_Tensor *input1 = RAI_ModelRunCtxInputTensor(op1->mctx, i);
        RAI_Tensor *input2 = RAI_ModelRunCtxInputTensor(op2->mctx, i);

        int ndims1 = RAI_TensorNumDims(input1);
        int ndims2 = RAI_TensorNumDims(input2);

        if (!RAI_TensorIsDataTypeEqual(input1, input2)) {
            return 0;
        }

        if (ndims1 != ndims2) {
            return 0;
        }

        if (ndims1 == 0) {
            continue;
        }

        for (int j = 1; j < ndims1; j++) {
            int dim1 = RAI_TensorDim(input1, j);
            int dim2 = RAI_TensorDim(input2, j);
            if (dim1 != dim2) {
                return 0;
            }
        }
    }

    return 1;
}

RAI_ModelRunCtx *RAI_GetAsModelRunCtx(RedisAI_RunInfo *rinfo, RAI_Error *err) {

    RAI_DagOp *op = rinfo->dagOps[0];
    if (!rinfo->single_op_dag || !op->mctx) {
        RAI_SetError(err, RedisAI_ErrorCode_EFINISHCTX, "Finish ctx is not a model run ctx");
        return NULL;
    }
    RAI_SetError(err, RAI_GetErrorCode(op->err), RAI_GetError(op->err));
    RAI_ModelRunCtx *mctx = op->mctx;
    rinfo->dagOps[0]->mctx = NULL;
    RAI_FreeRunInfo(rinfo);
    return mctx;
}

RAI_ScriptRunCtx *RAI_GetAsScriptRunCtx(RedisAI_RunInfo *rinfo, RAI_Error *err) {

    RAI_DagOp *op = rinfo->dagOps[0];
    if (!rinfo->single_op_dag || !op->sctx) {
        RAI_SetError(err, RedisAI_ErrorCode_EFINISHCTX, "Finish ctx is not a script run ctx");
        return NULL;
    }
    RAI_SetError(err, RAI_GetErrorCode(op->err), RAI_GetError(op->err));
    RAI_ScriptRunCtx *sctx = op->sctx;
    rinfo->dagOps[0]->sctx = NULL;
    RAI_FreeRunInfo(rinfo);
    return sctx;
}
