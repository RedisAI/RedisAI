/**
 * run_info.c
 *
 * Contains the methods to create, initialize, get, reset, and
 * free the structures that represent the context in which RedisAI blocking
 * commands operate, namely RedisAI_RunInfo and the newly added RAI_DagOp.
 *
 */

#include <pthread.h>
#include "redismodule.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "redis_ai_objects/script.h"
#include "redis_ai_objects/tensor.h"
#include "redis_ai_objects/model_struct.h"
#include "util/arr.h"
#include "util/dictionaries.h"
#include "util/string_utils.h"

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
 * Allocate the memory and initialise the RedisAI_RunInfo.
 * @param result Output parameter to capture allocated RedisAI_RunInfo.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitRunInfo(RedisAI_RunInfo **result) {
    RedisAI_RunInfo *rinfo;
    rinfo = (RedisAI_RunInfo *)RedisModule_Calloc(1, sizeof(RedisAI_RunInfo));

    rinfo->dagSharedTensors = array_new(RAI_Tensor *, 1);
    rinfo->persistTensors = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
    rinfo->tensorsNamesToIndices = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
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
    RedisModule_Assert(rinfo);
    long long ref_count = *rinfo->dagRefCount;
    RedisModule_Assert(ref_count == 0);
    pthread_rwlock_destroy(rinfo->dagLock);
    RedisModule_Free(rinfo->dagLock);

    size_t dag_tensors_num = array_len(rinfo->dagSharedTensors);
    for (size_t i = 0; i < dag_tensors_num; i++) {
        RAI_TensorFree(rinfo->dagSharedTensors[i]);
    }
    array_free(rinfo->dagSharedTensors);
    AI_dictRelease(rinfo->persistTensors);
    if (rinfo->tensorsNamesToIndices) {
        AI_dictRelease(rinfo->tensorsNamesToIndices);
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

RAI_ModelRunCtx *RAI_GetAsModelRunCtx(RedisAI_RunInfo *rinfo, RAI_Error *err) {

    RAI_DagOp *op = rinfo->dagOps[0];
    if (!rinfo->single_op_dag || !op->mctx) {
        RAI_SetError(err, RAI_EFINISHCTX, "Finish ctx is not a model run ctx");
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
        RAI_SetError(err, RAI_EFINISHCTX, "Finish ctx is not a script run ctx");
        return NULL;
    }
    RAI_SetError(err, RAI_GetErrorCode(op->err), RAI_GetError(op->err));
    RAI_ScriptRunCtx *sctx = op->sctx;
    rinfo->dagOps[0]->sctx = NULL;
    RAI_FreeRunInfo(rinfo);
    return sctx;
}
