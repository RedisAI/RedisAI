#include "model_script_run_session.h"

#include "batching.h"
#include "model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "script.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

size_t RAI_RunInfoBatchSize(struct RedisAI_RunInfo* rinfo) {
  if (rinfo->mctx == NULL) {
    return -1;
  }

  size_t ninputs = RAI_ModelRunCtxNumInputs(rinfo->mctx);

  int batchsize = 0;

  if (ninputs == 0) {
    return batchsize;
  }

  for (size_t i = 0; i < ninputs; i++) {
    RAI_Tensor* input = RAI_ModelRunCtxInputTensor(rinfo->mctx, 0, i);

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

int RAI_RunInfoBatchable(struct RedisAI_RunInfo* rinfo1,
                         struct RedisAI_RunInfo* rinfo2) {
  if (rinfo1->mctx == NULL || rinfo2->mctx == NULL) {
    return 0;
  }

  if (rinfo1->mctx->model != rinfo2->mctx->model) {
    return 0;
  }

  int ninputs1 = RAI_ModelRunCtxNumInputs(rinfo1->mctx);
  int ninputs2 = RAI_ModelRunCtxNumInputs(rinfo2->mctx);

  if (ninputs1 != ninputs2) {
    return 0;
  }

  for (int i = 0; i < ninputs1; i++) {
    RAI_Tensor* input1 = RAI_ModelRunCtxInputTensor(rinfo1->mctx, 0, i);
    RAI_Tensor* input2 = RAI_ModelRunCtxInputTensor(rinfo2->mctx, 0, i);

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

void RedisAI_FreeRunInfo(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo) {
  if (rinfo->mctx) {
    for (int i = 0; i < RAI_ModelRunCtxNumOutputs(rinfo->mctx); ++i) {
      RedisModule_FreeString(ctx, rinfo->outkeys[i]);
    }
    RedisModule_Free(rinfo->outkeys);
    RAI_ModelRunCtxFree(rinfo->mctx);
  } else if (rinfo->sctx) {
    for (int i = 0; i < RAI_ScriptRunCtxNumOutputs(rinfo->sctx); ++i) {
      RedisModule_FreeString(ctx, rinfo->outkeys[i]);
    }
    RedisModule_Free(rinfo->outkeys);
    RAI_ScriptRunCtxFree(rinfo->sctx);
  }

  if (rinfo->err) {
    RAI_ClearError(rinfo->err);
    RedisModule_Free(rinfo->err);
  }

  RedisModule_Free(rinfo);
}

/**
 * Actual method running the MODELRUN and SCRIPTRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RAI_ModelRunScriptRunSession(RedisAI_RunInfo **batch_rinfo) {
  if (array_len(batch_rinfo) == 0) {
    return NULL;
  }

  RAI_Error *err = RedisModule_Calloc(1, sizeof(RAI_Error));
  long long rtime;
  int status;
  RAI_ModelRunCtx *mctx = NULL;
  RAI_ScriptRunCtx *sctx = NULL;
  if (batch_rinfo[0]->mctx) {
    mctx = RAI_ModelRunCtxCreate(batch_rinfo[0]->mctx->model);
    for (long long i = 0; i < array_len(batch_rinfo); i++) {
      int id = RAI_ModelRunCtxAddBatch(mctx);
      RAI_ModelRunCtxCopyBatch(mctx, id, batch_rinfo[i]->mctx, 0);
    }
  } else if (batch_rinfo[0]->sctx) {
    // No batching for scripts for now
    sctx = batch_rinfo[0]->sctx;
  }

  const long long start = ustime();
  if (mctx) {
    status = RAI_ModelRun(mctx, err);
  } else if (sctx) {
    status = RAI_ScriptRun(sctx, err);
  }
  rtime = ustime() - start;

  for (long long i = 0; i < array_len(batch_rinfo); i++) {
    struct RedisAI_RunInfo *rinfo = batch_rinfo[i];
    if (mctx) {
      size_t noutputs = RAI_ModelRunCtxNumOutputs(mctx);
      for (long long o = 0; o < noutputs; o++) {
        RAI_Tensor *tensor = mctx->batches[i].outputs[o].tensor;
        if (tensor) {
          rinfo->mctx->batches[0].outputs[o].tensor =
              RAI_TensorGetShallowCopy(tensor);
        } else {
          rinfo->mctx->batches[0].outputs[o].tensor = NULL;
        }
      }
    } else if (sctx) {
      // No batching for scripts for now
    }

    rinfo->status = status;
    rinfo->err = RedisModule_Calloc(1, sizeof(RAI_Error));
    // TODO: add information on whether the call was batched
    // and how large the batch was
    rinfo->duration_us = rtime;

    rinfo->err->code = err->code;
    if (err->code != RAI_OK) {
      rinfo->err->detail = RedisModule_Strdup(err->detail);
      rinfo->err->detail_oneline = RedisModule_Strdup(err->detail_oneline);
    }
    if (rinfo->client != NULL) {
      RedisModule_UnblockClient(rinfo->client, rinfo);
    }
  }

  if (mctx) {
    RAI_ModelRunCtxFree(mctx);
  } else if (sctx) {
    // No batching for scripts for now
  }

  return NULL;
}

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() within
 * RAI_ModelRunScriptRunSession() in order to reply to the client and unblock it
 */
int RAI_ModelRunScriptRunReply(RedisModuleCtx *ctx, RedisModuleString **argv,
                               int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  struct RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

  const char *runkey = RedisModule_StringPtrLen(rinfo->runkey, NULL);
  AI_dictEntry *stats_entry = AI_dictFind(run_stats, runkey);

  struct RedisAI_RunStats *rstats = NULL;
  if (stats_entry) {
    rstats = AI_dictGetVal(stats_entry);
  }

  if (rinfo->status) {
    RedisModule_Log(ctx, "warning", "ERR %s", rinfo->err->detail);
    if (rstats) {
      rstats->calls += 1;
      rstats->nerrors += 1;
    }
    int ret = RedisModule_ReplyWithError(ctx, rinfo->err->detail_oneline);
    RedisAI_FreeRunInfo(ctx, rinfo);
    return ret;
  }

  size_t num_outputs = 0;
  if (rinfo->mctx) {
    num_outputs = RAI_ModelRunCtxNumOutputs(rinfo->mctx);
  } else if (rinfo->sctx) {
    num_outputs = RAI_ScriptRunCtxNumOutputs(rinfo->sctx);
  }

  int64_t batch_size = 0;

  for (size_t i = 0; i < num_outputs; ++i) {
    RedisModuleKey *outkey;
    const int status = RAI_OpenKey_Tensor(ctx, rinfo->outkeys[i], &outkey,
                                          REDISMODULE_READ | REDISMODULE_WRITE);
    if (status == REDISMODULE_ERR) {
      RedisAI_FreeRunInfo(ctx, rinfo);
      if (rstats) {
        rstats->calls += 1;
        rstats->nerrors += 1;
      }
      return REDISMODULE_ERR;
    }
    RAI_Tensor *t = NULL;
    if (rinfo->mctx) {
      t = RAI_ModelRunCtxOutputTensor(rinfo->mctx, 0, i);
      if (t && batch_size == 0) {
        batch_size = RAI_TensorDim(t, 0);
      }
    } else if (rinfo->sctx) {
      t = RAI_ScriptRunCtxOutputTensor(rinfo->sctx, i);
    }
    if (t) {
      RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType,
                                     RAI_TensorGetShallowCopy(t));
    }
    RedisModule_CloseKey(outkey);

    if (t) {
      RedisAI_ReplicateTensorSet(ctx, rinfo->outkeys[i], t);
    }
  }

  if (rstats) {
    rstats->duration_us += rinfo->duration_us;
    rstats->calls += 1;

    if (rinfo->mctx) {
      rstats->samples += batch_size;
    }
  }

  // FIXME This crashes Redis, we need to investigate.
  // RedisModule_CloseKey(rinfo->modelkey);

  RedisAI_FreeRunInfo(ctx, rinfo);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/**
 * Called in order to free the private data that is passed
 * by RedisModule_UnblockClient() call after
 * RAI_ModelRunScriptRunSession()
 */
void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo) {}

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
  RedisModule_Log(ctx, "warning", "Blocked client %p disconnected!",
                  (void *)bc);
}