#ifndef SRC_MODEL_SCRIPT_RUN_SESSION_H_
#define SRC_MODEL_SCRIPT_RUN_SESSION_H_

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

size_t RAI_RunInfoBatchSize(struct RedisAI_RunInfo* rinfo);
int RAI_RunInfoBatchable(struct RedisAI_RunInfo* rinfo1, struct RedisAI_RunInfo* rinfo2);

/**
 * Actual method running the MODELRUN and SCRIPTRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RAI_ModelRunScriptRunSession(RedisAI_RunInfo **batch_rinfo);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() within
 * RAI_ModelRunScriptRunSession() in order to reply to the client and unblock it
 */
int RAI_ModelRunScriptRunReply(RedisModuleCtx *ctx, RedisModuleString **argv,
                               int argc);

/**
 * Called in order to free the private data that is passed
 * by RedisModule_UnblockClient() call after
 * RAI_ModelRunScriptRunSession()
 */
void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo);

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc);

void RedisAI_FreeRunInfo(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo);

#endif /* SRC_MODEL_SCRIPT_RUN_SESSION_H_ */
