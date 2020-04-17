#include "err.h"
#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"

/**
 * Allocate the memory and initialise the RAI_DagOp.
 * @param result Output parameter to capture allocated RAI_DagOp.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the allocation
 * failed.
 */
int RAI_InitDagOp(RAI_DagOp **result) {
  RAI_DagOp *dagOp;
  dagOp = (RAI_DagOp *)RedisModule_Calloc(1, sizeof(RAI_DagOp));
  if (!dagOp) {
    return REDISMODULE_ERR;
  }
  dagOp->commandType = REDISAI_DAG_CMD_NONE;
  dagOp->runkey = NULL;
  dagOp->outkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
  if (!(dagOp->outkeys)) {
    return REDISMODULE_ERR;
  }
  dagOp->outTensors = (RAI_Tensor *)array_new(RAI_Tensor, 1);
  if (!(dagOp->outTensors)) {
    return REDISMODULE_ERR;
  }
  dagOp->mctx = NULL;
  dagOp->sctx = NULL;
  dagOp->duration_us = 0;
  RAI_InitError(&dagOp->err);
  if (!(dagOp->err)) {
    return REDISMODULE_ERR;
  }
  dagOp->argv = (RedisModuleString **)array_new(RedisModuleString *, 1);
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
int RAI_InitRunInfo(RedisAI_RunInfo **result) {
  RedisAI_RunInfo *rinfo;
  rinfo = (RedisAI_RunInfo *)RedisModule_Calloc(1, sizeof(RedisAI_RunInfo));
  if (!rinfo) {
    return REDISMODULE_ERR;
  }
  rinfo->runkey = NULL;
  rinfo->outkeys = (RedisModuleString **)array_new(RedisModuleString *, 1);
  rinfo->mctx = NULL;
  rinfo->sctx = NULL;
  rinfo->duration_us = 0;
  RAI_InitError(&rinfo->err);
  if (!(rinfo->err)) {
    return REDISMODULE_ERR;
  }
  rinfo->use_local_context = 0;
  rinfo->dagTensorsContext = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  if (!(rinfo->dagTensorsContext)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagTensorsPersistentContext =
      AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  if (!(rinfo->dagTensorsPersistentContext)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagOps = (RAI_DagOp **)array_new(RAI_DagOp *, 1);
  if (!(rinfo->dagOps)) {
    return REDISMODULE_ERR;
  }
  rinfo->dagReplyLength = 0;
  rinfo->dagNumberCommands = 0;
  *result = rinfo;
  return REDISMODULE_OK;
}

void RAI_FreeDagOp(RedisModuleCtx *ctx, RAI_DagOp *dagOp) {
  if (dagOp) {
    RAI_FreeError(dagOp->err);
    if (dagOp->argv) {
      for (size_t i = 0; i < array_len(dagOp->argv); i++) {
        RedisModule_FreeString(ctx, dagOp->argv[i]);
      }
      array_free(dagOp->argv);
    }
    // dagOp->outkeys is released on all argv release above
    // dagOp->outTensors is released on RunInfo after checking what tensors to
    // persist
    for (size_t i = 0; i < array_len(dagOp->outTensors); i++) {
      RAI_TensorFree(dagOp->outTensors[i]);
    }
    array_free(dagOp->outTensors);

    RedisModule_Free(dagOp);
  }
}

void RAI_FreeRunInfo(RedisModuleCtx *ctx, struct RedisAI_RunInfo *rinfo) {
  if (rinfo->mctx) {
    RAI_ModelRunCtxFree(rinfo->mctx);
  }
  if (rinfo->sctx) {
    RAI_ScriptRunCtxFree(rinfo->sctx);
  }
  RAI_FreeError(rinfo->err);

  if (rinfo->dagTensorsContext) {
    AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsContext);
    AI_dictEntry *stats_entry = AI_dictNext(iter);
    RAI_Tensor *tensor = NULL;

    while (stats_entry) {
      tensor = AI_dictGetVal(stats_entry);
      const char *key = (char *)AI_dictGetKey(stats_entry);

      if (tensor) {
        // if the key is persistent then we should not delete it
        AI_dictEntry *persistent_entry =
            AI_dictFind(rinfo->dagTensorsPersistentContext, key);
        if (!persistent_entry) {
          RAI_TensorFree(tensor);
        }
      }
      RedisModule_Free(key);
      stats_entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
    RedisModule_Free(rinfo->dagTensorsContext);
  }

  if (rinfo->dagTensorsPersistentContext) {
    AI_dictIterator *iter =
        AI_dictGetSafeIterator(rinfo->dagTensorsPersistentContext);
    AI_dictEntry *stats_entry = AI_dictNext(iter);
    while (stats_entry) {
      const char *key = (char *)AI_dictGetKey(stats_entry);
      RedisModule_Free(key);
      stats_entry = AI_dictNext(iter);
    }
    AI_dictReleaseIterator(iter);
    RedisModule_Free(rinfo->dagTensorsPersistentContext);
  }

  if (rinfo->dagOps) {
    for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
      RAI_FreeDagOp(ctx, rinfo->dagOps[i]);
    }
    array_free(rinfo->dagOps);
  }

  if (rinfo->outkeys) {
    for (size_t i = 0; i < array_len(rinfo->outkeys); i++) {
      RedisModule_FreeString(ctx, rinfo->outkeys[i]);
    }
    array_free(rinfo->outkeys);
  }

  RedisModule_Free(rinfo);
}

size_t RAI_RunInfoBatchSize(struct RedisAI_RunInfo *rinfo) {
  if (rinfo->mctx == NULL) {
    return -1;
  }

  size_t ninputs = RAI_ModelRunCtxNumInputs(rinfo->mctx);

  int batchsize = 0;

  if (ninputs == 0) {
    return batchsize;
  }

  for (size_t i = 0; i < ninputs; i++) {
    RAI_Tensor *input = RAI_ModelRunCtxInputTensor(rinfo->mctx, i);

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

int RAI_RunInfoBatchable(struct RedisAI_RunInfo *rinfo1,
                         struct RedisAI_RunInfo *rinfo2) {
  if (rinfo1->mctx == NULL || rinfo2->mctx == NULL) {
    return 0;
  }

  if (rinfo1->mctx->model != rinfo2->mctx->model) {
    return 0;
  }

  const int ninputs1 = RAI_ModelRunCtxNumInputs(rinfo1->mctx);
  const int ninputs2 = RAI_ModelRunCtxNumInputs(rinfo2->mctx);

  if (ninputs1 != ninputs2) {
    return 0;
  }

  for (int i = 0; i < ninputs1; i++) {
    RAI_Tensor *input1 = RAI_ModelRunCtxInputTensor(rinfo1->mctx, i);
    RAI_Tensor *input2 = RAI_ModelRunCtxInputTensor(rinfo2->mctx, i);

    int ndims1 = RAI_TensorNumDims(input1);
    int ndims2 = RAI_TensorNumDims(input2);

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