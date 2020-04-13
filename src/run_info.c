#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "util/dict.h"
#include "util/arr_rm_alloc.h"

/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int dagInit(RAI_DagOp** result) {
  *result = RedisModule_Calloc(1, sizeof(RAI_DagOp*));
  if (!(*result)) {
    return REDISMODULE_ERR;
  }
  (*result)->runkey = NULL;
  (*result)->outkeys = array_new(RedisModuleString*, 10);
  if (!((*result)->outkeys)) {
    return REDISMODULE_ERR;
  }
  (*result)->mctx = NULL;
  (*result)->sctx = NULL;
  (*result)->duration_us = 0;
  (*result)->err = NULL;
  (*result)->argv = array_new(RedisModuleString*, 10);
  if (!((*result)->argv)) {
    return REDISMODULE_ERR;
  }
  (*result)->argc = 0;
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
  rinfo = (RedisAI_RunInfo*)calloc(1, sizeof(RedisAI_RunInfo));
  if (!rinfo) {
    return REDISMODULE_ERR;
  }
  rinfo->runkey = NULL;
  rinfo->outkeys = NULL;
  rinfo->mctx = NULL;
  rinfo->sctx = NULL;
  rinfo->duration_us = 0;
  rinfo->err = NULL;
  rinfo->use_local_context = 0;
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