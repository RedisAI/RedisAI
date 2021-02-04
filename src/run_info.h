/**
 * run_info.h
 *
 * Contains the structure and headers to create, initialize, get, reset, and
 * free the structures that represent the context in which RedisAI blocking
 * commands operate, namely RedisAI_RunInfo and the newly added RAI_DagOp.
 *
 */

#ifndef SRC_RUN_INFO_H_
#define SRC_RUN_INFO_H_

#include "err.h"
#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"

typedef enum DAGCommand {
    REDISAI_DAG_CMD_NONE = 0,
    REDISAI_DAG_CMD_TENSORSET,
    REDISAI_DAG_CMD_TENSORGET,
    REDISAI_DAG_CMD_MODELRUN,
    REDISAI_DAG_CMD_SCRIPTRUN
} DAGCommand;

enum RedisAI_DAGMode { REDISAI_DAG_READONLY_MODE = 0, REDISAI_DAG_WRITE_MODE };

typedef struct RAI_DagOp {
    int commandType;
    RedisModuleString *runkey;
    RedisModuleString **inkeys;
    RedisModuleString **outkeys;
    size_t *inkeys_indices;
    size_t *outkeys_indices;
    RAI_Tensor *outTensor; // The tensor to upload in TENSORSET op.
    RAI_ModelRunCtx *mctx;
    RAI_ScriptRunCtx *sctx;
    uint fmt; // This is relevant for TENSORGET op.
    char *devicestr;
    int result; // REDISMODULE_OK or REDISMODULE_ERR
    long long duration_us;
    RAI_Error *err;
    RedisModuleString **argv;
    int argc;
} RAI_DagOp;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitDagOp(RAI_DagOp **result);

/**
 * Frees the memory allocated of RAI_DagOp
 * @param ctx Context in which Redis modules operate
 * @param RAI_DagOp context in which RedisAI command operates.
 */
void RAI_FreeDagOp(RAI_DagOp *dagOp);

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
 * Obtain the batch size for the provided DAG operation, that is, the
 * size of the tensor in the zero-th dimension
 * @param op DAG operation to operate on
 * @return size of the batch for op
 */
size_t RAI_RunInfoBatchSize(struct RAI_DagOp *op);

/**
 * Find out whether two DAG operations are batchable. That means they must be
 * two MODELRUN operations with the same model, where respective inputs have
 * compatible shapes (all dimensions except the zero-th must match)
 * @param op1 first DAG operation
 * @param op2 second DAG operation
 * @return 1 if batchable, 0 otherwise
 */
int RAI_RunInfoBatchable(struct RAI_DagOp *op1, struct RAI_DagOp *op2);

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

#endif /* SRC_RUN_INFO_H_ */
