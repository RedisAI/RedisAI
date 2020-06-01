/**
 * dag.h
 *
 * Contains headers for the helper methods for both parsing, running the command in the
 * background, and replying DAG structured commands.
 */

#ifndef SRC_DAG_H_
#define SRC_DAG_H_

#include "model.h"
#include "redisai.h"
#include "run_info.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"

/**
 * Actual method running the DAGRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 * After all computation is done, this will trigger
 * the reply callback to be called in order to reply to the client.
 * The 'rinfo' argument will be accessible by the reply callback.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @return
 */
void *RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char* devicestr, int *progress, int *complete);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() within
 * RedisAI_DagRunSession() in order to reply to the client and unblock it
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc);

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param loadedContextDict local non-blocking hash table containing key names
 * loaded from the keyspace tensors
 * @param localContextDict local non-blocking hash table containing DAG's
 * tensors
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc, AI_dict **loadedContextDict,
                         AI_dict **localContextDict,
                         const char *chaining_operator);

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param localContextDict local non-blocking hash table containing DAG's
 * keynames marked as persistent
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                            int argc, AI_dict **localContextDict,
                            const char *chaining_operator);

/**
 * DAGRUN and DAGRUN_RO parser, which reads the the sequence of
 * arguments and decides whether the sequence conforms to the syntax
 * specified by the DAG grammar.
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param dagMode access mode, for now REDISAI_DAG_READONLY_MODE or REDISAI_DAG_WRITE_MODE
 * @return
 */
int RedisAI_DagRunSyntaxParser(RedisModuleCtx *ctx, RedisModuleString **argv,
                                 int argc, int dagMode);

#endif /* SRC_DAG_H_ */
