/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/**
 * run_info.h
 *
 * Contains the structure and headers to create, initialize, get, reset, and
 * free the structures that represent the context in which RedisAI blocking
 * commands operate, namely RedisAI_RunInfo and the newly added RAI_DagOp.
 *
 */

#pragma once

#include "redismodule.h"
#include "redis_ai_objects/err.h"
#include "execution/DAG/dag_op.h"
#include "util/arr.h"
#include "util/dict.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RedisAI_RunInfo RedisAI_RunInfo;

/**
 * This structure contains the context data at the end of the execution.
 * user can access results and errors through LLAPI.
 */
typedef RedisAI_RunInfo RedisAI_OnFinishCtx;

/**
 * @brief User defined callback to execute at the end of the run.
 * @param ctx parameter includes the running results and errors.
 * @param private_data is an optional pointer to the user's private data.
 */
typedef void (*RedisAI_OnFinishCB)(RedisAI_OnFinishCtx *ctx, void *private_data);

/**
 * This structure represents the context in which RedisAI blocking commands
 * operate.
 *
 * Note that not all the context structure is always filled with actual values
 * but only the fields needed in a given operation.
 */

struct RedisAI_RunInfo {
    RedisModuleBlockedClient *client;
    int single_op_dag;
    int single_device_dag;
    RAI_Tensor **dagSharedTensors;  // Shared array of tensors that dag ops use.
    AI_dict *persistTensors;        // Associates the tensors to persist with their indices .
    AI_dict *tensorsNamesToIndices; // Maps tensor key name to its (maximal) index.
    RAI_DagOp **dagOps;             // all ops in DAG
    RAI_DagOp **dagDeviceOps;       // all ops in DAG for device
    int dagReplyLength;
    int dagOpCount;               // number of ops in DAG
    int *dagCompleteOpCount;      // number of completed ops in DAG
    int dagDeviceOpCount;         // number of ops in DAG for device
    int dagDeviceCompleteOpCount; // number of completed ops in DAG for device
    // Pointer to integer signaling whether an error occurred anywhere in the DAG.
    // This is shared across shallow copies in device queues.
    int *dagError;
    // DAG global error.
    RAI_Error *err;
    // Pointer to mutex used to exclusively access DagOps from multiple worker threads.
    pthread_rwlock_t *dagLock;
    // Pointer to ref count in DAG, shared across multiple worker thread
    long long *dagRefCount;
    long long timeout;
    int *timedOut;
    struct timeval queuingTime;
    RedisAI_OnFinishCB OnFinish;
    RedisAI_RunInfo *orig_copy;
    void *private_data; // This is going to be sent to the OnFinish callback.
};

/**
 * Allocate the memory and initialise the RedisAI_RunInfo.
 * @param result Output parameter to capture allocated RedisAI_RunInfo.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitRunInfo(RedisAI_RunInfo **result);

int RAI_ShallowCopyDagRunInfo(RedisAI_RunInfo **result, RedisAI_RunInfo *src);

/**
 * Frees the shallow copy of RedisAI_RunInfo pointed by rinfo.
 * @param rinfo copy to be freed.
 * @retval The ref_count of the rinfo object after freeing this copy.
 */
long long RAI_DagRunInfoFreeShallowCopy(RedisAI_RunInfo *rinfo);

/**
 * Frees the memory allocated on RedisAI_RunInfo
 * @param ctx Context in which Redis modules operate
 * @param rinfo context in which RedisAI blocking command operate.
 */
void RAI_FreeRunInfo(RedisAI_RunInfo *rinfo);

/**
 * Locks the DAG tensor context rwlock for reads. No-op in case of single
 * op or single device DAGS.
 * @param rinfo context in which RedisAI blocking command operate.
 */
void RAI_ContextReadLock(RedisAI_RunInfo *rinfo);

/**
 * Locks the DAG tensor context rwlock for writes. No-op in case of single
 * op or single device DAGS.
 * @param rinfo context in which RedisAI blocking command operate.
 */
void RAI_ContextWriteLock(RedisAI_RunInfo *rinfo);

/**
 * Unlocks the DAG tensor context rwlock. No-op in case of single op or single
 * device DAGS.
 * @param rinfo context in which RedisAI blocking command operate.
 */
void RAI_ContextUnlock(RedisAI_RunInfo *rinfo);

/**
 * Retreive the ModelRunCtx of a DAG runInfo that contains a single op of type
 * MODELRUN.
 * @param DAG runInfo.
 * @return Pointer to the ModelRunCtx in DAG's single op.
 */
RAI_ModelRunCtx *RAI_GetAsModelRunCtx(RedisAI_RunInfo *rinfo, RAI_Error *err);

/**
 * Retreive the ScriptRunCtx of a DAG runInfo that contains a single op of type
 * SCRIPTRUN.
 * @param DAG runInfo.
 * @return Pointer to the ScriptRunCtx in DAG's single op.
 */
RAI_ScriptRunCtx *RAI_GetAsScriptRunCtx(RedisAI_RunInfo *rinfo, RAI_Error *err);

#ifdef __cplusplus
} // extern "C"
#endif
