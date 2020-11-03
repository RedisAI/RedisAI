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

enum RedisAI_DAGCommands {
  REDISAI_DAG_CMD_NONE = 0,
  REDISAI_DAG_CMD_TENSORSET,
  REDISAI_DAG_CMD_TENSORGET,
  REDISAI_DAG_CMD_MODELRUN,
  REDISAI_DAG_CMD_SCRIPTRUN
};

enum RedisAI_DAGMode {
    REDISAI_DAG_READONLY_MODE = 0,
    REDISAI_DAG_WRITE_MODE
};

typedef struct RAI_DagOp {
  int commandType;
  RedisModuleString *runkey;
  RedisModuleString **inkeys;
  RedisModuleString **outkeys;
  RAI_Tensor **outTensors;
  RAI_ModelRunCtx *mctx;
  RAI_ScriptRunCtx *sctx;
  char* devicestr;
  int result;  // REDISMODULE_OK or REDISMODULE_ERR
  long long duration_us;
  RAI_Error *err;
  RedisModuleString **argv;
  int argc;
} RAI_DagOp;

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
void RAI_FreeDagOp(RedisModuleCtx *ctx, RAI_DagOp *dagOp);

/**
 * This structure represents the context in which RedisAI blocking commands
 * operate.
 *
 * Note that not all the context structure is always filled with actual values
 * but only the fields needed in a given operation.
 */
typedef struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  int single_op_dag;
  int single_device_dag;
  AI_dict *dagTensorsContext;
  AI_dict *dagTensorsPersistedContext;  // dict to flag tensors to persist
  AI_dict *dagTensorsLoadedContext;  // dict to flag tensors loaded from the keyspace
  RAI_DagOp **dagOps;
  int dagReplyLength;
  int dagNumberCommands;
  // Pointer to integer signaling whether an error occurred anywhere in the DAG.
  // This is shared across shallow copies in device queues.
  int *dagError;
  // Pointer to mutex used to exclusively access DagOps from multiple worker threads.
  pthread_mutex_t *dagMutex;
  int dagMaster;
  // Pointer to ref count in DAG, shared across multiple worker thread
  long long *dagRefCount;
} RedisAI_RunInfo;

/**
 * Allocate the memory and initialise the RedisAI_RunInfo.
 * @param result Output parameter to capture allocated RedisAI_RunInfo.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitRunInfo(RedisAI_RunInfo **result);

int RAI_ShallowCopyDagRunInfo(RedisAI_RunInfo **result, RedisAI_RunInfo *src);
 
/**
 * Frees the memory allocated on RedisAI_RunInfo
 * @param ctx Context in which Redis modules operate
 * @param rinfo context in which RedisAI blocking command operate.
 */
void RAI_FreeRunInfo(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo);

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
int RAI_RunInfoBatchable(struct RAI_DagOp *op1,
                         struct RAI_DagOp *op2);

#endif /* SRC_RUN_INFO_H_ */
