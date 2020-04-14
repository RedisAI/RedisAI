

#ifndef SRC_RUN_INFO_H_
#define SRC_RUN_INFO_H_

#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "util/dict.h"
#include "util/arr_rm_alloc.h"
#include "err.h"

enum RedisAI_DAGCommands {
  REDISAI_DAG_CMD_NONE = 0,
  REDISAI_DAG_CMD_TENSORSET,
  REDISAI_DAG_CMD_TENSORGET,
  REDISAI_DAG_CMD_MODELRUN
};

typedef struct RAI_DagOp {
  int commandType;
  RedisModuleString *runkey;
  RedisModuleString **outkeys;
  RAI_Tensor **outTensors;
  RAI_ModelRunCtx *mctx;
  RAI_ScriptRunCtx *sctx;
  int result; // REDISMODULE_OK or REDISMODULE_ERR
  long long duration_us;
  RAI_Error* err;
  RedisModuleString **argv;
  int argc;
} RAI_DagOp;

/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int dagInit(RAI_DagOp **result);

typedef struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  // TODO: completly move modelrun and scriptrun to dagOps
  RedisModuleString *runkey;
  RedisModuleString **outkeys;
  RAI_ModelRunCtx *mctx;
  RAI_ScriptRunCtx *sctx;
  int result; // REDISMODULE_OK or REDISMODULE_ERR
  long long duration_us;
  RAI_Error *err;
  // DAG
  int use_local_context;
  AI_dict *dagTensorsContext;
  AI_dict *dagTensorsPersistentContext;
  RAI_DagOp **dagOps;
  int dagReplyLength;
  int dagNumberCommands;
} RedisAI_RunInfo;

/**
 * Allocate the memory and initialise the RedisAI_RunInfo.
 * @param result Output parameter to capture allocated RedisAI_RunInfo.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int runInfoInit(RedisAI_RunInfo **result);

#endif /* SRC_RUN_INFO_H_ */
