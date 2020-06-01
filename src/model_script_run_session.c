/**
 * model_scrip_run_session.c
 * 
 * Contains the methods that are related to async background work that was
 * triggered by either MODELRUN or SCRIPTRUN Command and Called within
 * `RedisAI_Run_ThreadMain`. This file also contains the function signatures of
 * the reply callbacks to be called in order to reply to the clients, after the
 * background work on MODELRUN and SCRIPTRUN is done.
 *
 */

#include "model_script_run_session.h"

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

/**
 * Actual method running the MODELRUN and SCRIPTRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RAI_ModelRunScriptRunSession(RedisAI_RunInfo **batch_rinfo) {
  const long long batch_size = array_len(batch_rinfo);

  if (batch_size == 0) {
    return NULL;
  }

  RAI_ScriptRunCtx *sctx = NULL;

  // RAI_Error *err = RedisModule_Calloc(1, sizeof(RAI_Error));
  RAI_Error err = {0};
  long long rtime;
  int status;
  bool is_model;
  if (batch_rinfo[0]->mctx) {
    RAI_ModelRunCtx *mctxs[batch_size];
    for (long long i = 0; i < batch_size; i++) {
      mctxs[i] = batch_rinfo[i]->mctx;
    }
    const long long start = ustime();
    status = RAI_ModelRun(mctxs, batch_size, &err);
    rtime = ustime() - start;
  } else if (batch_rinfo[0]->sctx) {
    sctx = batch_rinfo[0]->sctx;
    const long long start = ustime();
    status = RAI_ScriptRun(sctx, &err);
    rtime = ustime() - start;
  }

  for (long long i = 0; i < batch_size; i++) {
    struct RedisAI_RunInfo *rinfo = batch_rinfo[i];

    rinfo->result = status;
    rinfo->err = RedisModule_Calloc(1, sizeof(RAI_Error));
    // TODO: add information on whether the call was batched
    // and how large the batch was
    rinfo->duration_us = rtime;

    rinfo->err->code = err.code;
    if (err.code != RAI_OK) {
      rinfo->err->detail = RedisModule_Strdup(err.detail);
      rinfo->err->detail_oneline = RedisModule_Strdup(err.detail_oneline);
    }
    if (rinfo->client != NULL) {
      RedisModule_UnblockClient(rinfo->client, rinfo);
    }
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
  struct RedisAI_RunStats *rstats = NULL;
  RAI_GetRunStats(runkey, &rstats);

  if (rinfo->result == REDISMODULE_ERR) {
    RedisModule_Log(ctx, "warning", "ERR %s", rinfo->err->detail);
    int ret = RedisModule_ReplyWithError(ctx, rinfo->err->detail_oneline);
    RAI_SafeAddDataPoint(rstats, 0, 1, 1, 0);
    RAI_FreeRunInfo(ctx, rinfo);
    return REDISMODULE_ERR;
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
      RAI_FreeRunInfo(ctx, rinfo);
      RAI_SafeAddDataPoint(rstats, 0, 1, 1, 0);
      return REDISMODULE_ERR;
    }
    RAI_Tensor *t = NULL;
    if (rinfo->mctx) {
      t = RAI_ModelRunCtxOutputTensor(rinfo->mctx, i);
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
  RAI_SafeAddDataPoint(rstats, rinfo->duration_us, 1, 0, batch_size);
  RAI_FreeRunInfo(ctx, rinfo);
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
