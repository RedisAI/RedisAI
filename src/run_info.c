#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "err.h"
/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int dagInit(RAI_DagOp** result) {
  RAI_DagOp* dagOp;
  dagOp = (RAI_DagOp*)RedisModule_Calloc(1, sizeof(RAI_DagOp));
  if (!dagOp) {
    return REDISMODULE_ERR;
  }
  dagOp->commandName = NULL;
  dagOp->runkey = NULL;
  dagOp->outkeys = (RedisModuleString**)array_new(RedisModuleString*, 10);
  if (!(dagOp->outkeys)) {
    return REDISMODULE_ERR;
  }
  dagOp->mctx = NULL;
  dagOp->sctx = NULL;
  dagOp->duration_us = 0;
  RAI_InitError(&dagOp->err);
  if (!(dagOp->err)) {
    return REDISMODULE_ERR;
  }
  dagOp->argv = (RedisModuleString**)array_new(RedisModuleString*, 10);
  if (!(dagOp->argv)) {
    return REDISMODULE_ERR;
  }
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
int runInfoInit(RedisAI_RunInfo** result) {
  RedisAI_RunInfo* rinfo;
  rinfo = (RedisAI_RunInfo*)RedisModule_Calloc(1, sizeof(RedisAI_RunInfo));
  if (!rinfo) {
    return REDISMODULE_ERR;
  }
  rinfo->runkey = NULL;
  rinfo->outkeys = NULL;
  rinfo->mctx = NULL;
  rinfo->sctx = NULL;
  rinfo->duration_us = 0;
  RAI_InitError(&rinfo->err);
  if (!(rinfo->err)) {
    return REDISMODULE_ERR;
  }
  rinfo->use_local_context = 0;
  rinfo->dagTensorsContext =
      AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  if (!(rinfo->dagTensorsContext)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagTensorsPersistentContext =
      AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  if (!(rinfo->dagTensorsPersistentContext)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagOps = (RAI_DagOp**)array_new(RAI_DagOp*, 10);
  if (!(rinfo->dagOps)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagReplyLength = 0;
  rinfo->dagNumberCommands = 0;
  *result = rinfo;
  return REDISMODULE_OK;
}