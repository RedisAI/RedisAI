#ifndef SRC_DAG_H_
#define SRC_DAG_H_


#include "model.h"
#include "redisai.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "run_info.h"

/**
 * Actual method running the DAGRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RedisAI_DagRunSession(RedisAI_RunInfo *rinfo);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() within
 * RedisAI_RunDag() in order to reply to the client and unblock it
 */
int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc);

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc, AI_dict **localContextDict,
                         const char *chaining_operator);

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                            int argc, AI_dict **localContextDict,
                            const char *chaining_operator);

#endif /* SRC_DAG_H_ */
