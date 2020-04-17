#ifndef SRC_MODEL_SCRIPT_RUN_SESSION_H_
#define SRC_MODEL_SCRIPT_RUN_SESSION_H_

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
 * After all computation is done, this will trigger
 * the reply callbacks to be called in order to reply to the clients.
 * The 'rinfo' argument will be accessible by the reply callback, for each of
 * the runinfo present in batch_rinfo
 *
 * @param batch_rinfo array of `RedisAI_RunInfo *rinfo` contexts in which RedisAI blocking commands operate.
 * @return
 */
void *RAI_ModelRunScriptRunSession(RedisAI_RunInfo **batch_rinfo);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() within
 * RAI_ModelRunScriptRunSession() in order to reply to the client and unblock it
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the MODELRUN/SCRIPTRUN failed
 */
int RAI_ModelRunScriptRunReply(RedisModuleCtx *ctx, RedisModuleString **argv,
                               int argc);

/**
 * Called in order to free the private data that is passed
 * by RedisModule_UnblockClient() call after
 * RAI_ModelRunScriptRunSession()
 *
 * @param ctx Context in which Redis modules operate
 * @param rinfo
 */
void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo);

/**
 *
 * @param ctx Context in which Redis modules operate
 * @param bc
 */
void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc);


#endif /* SRC_MODEL_SCRIPT_RUN_SESSION_H_ */
