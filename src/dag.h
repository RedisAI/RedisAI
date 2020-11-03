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
 * Get current DAG op for the given device. An op is current if it's
 * the first unrealized op for the device.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @param currentOp current op identified
 * @return
 */
void RedisAI_DagCurrentOp(RedisAI_RunInfo *rinfo, const char *devicestr,
                          RAI_DagOp **currentOp);
 
/**
 * Get current DAG op for the given device. An op is current if it's
 * the first unrealized op for the device. Also get additional information
 * about the op.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @param currentOp current op identified
 * @param currentOpReady have all inputs for the op been computed, that is,
 *            are they available in the tensor context
 * @param currentOpBatchable is the op amenable to batching, that is, is it
 *            a MODELRUN and is BATCHSIZE greater than zero
 * @param deviceComplete were all ops for the given device already computed
 * @param dagComplete were all ops in the DAG already computed
 * @return
 */
void RedisAI_DagCurrentOpAndInfo(RedisAI_RunInfo *rinfo, const char *devicestr,
                                 RAI_DagOp **currentOp, int *currentOpReady,
                                 int *currentOpBatchable,
                                 int *deviceComplete, int *dagComplete);

/**
 * Get batching information about a DAG op.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param op DAG operation
 * @param batchsize maximum batch size specified by BATCHSIZE
 * @param minbatchsize minimum batch size specified by MINBATCHSIZE
 * @param inbatchsize actual size of the batch in the current op, that
 *            is, the size of the input tensors along the zero-th dimension
 * @return
 */
void RedisAI_DagOpBatchInfo(RedisAI_RunInfo *rinfo, RAI_DagOp *op,
                            size_t *batchsize, size_t *minbatchsize,
                            size_t *inbatchsize);
 
/**
 * Check that a DAG operation can be batched with a given batch operation.
 * @param rinfo1 given context in which RedisAI blocking commands operate.
 * @param op1 given DAG operation
 * @param rinfo2 other context in which RedisAI blocking commands operate.
 * @param opr other DAG operation to be checked
 * @param batched can op2 be batched with op1
 * @param inbatchsize actual size of the batch in op2
 * @return
 */
void RedisAI_DagOpBatchingMatch(RedisAI_RunInfo *rinfo1, RAI_DagOp *op1,
                                RedisAI_RunInfo *rinfo2, RAI_DagOp *op2,
                                int *batched, size_t *inbatchsize);
 
/**
 * Run the first unrealized DAG operation in rinfo for the given device.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @return
 */
void RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char* devicestr);

/**
 * Batch the first unrealized DAG operations for the given device for the
 * provided rinfo and run.
 * @param rinfo contexts in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @return
 */
void RedisAI_BatchedDagRunSessionStep(RedisAI_RunInfo **rinfo, const char* devicestr);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() after
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
 * When a module command is called in order to obtain the position of
 * keys, since it was flagged as "getkeys-api" during the registration,
 * the command implementation checks for this special call using the
 * RedisModule_IsKeysPositionRequest() API and uses this function in
 * order to report keys.
 * No real execution is done on this special call.
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return
 */
int RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                            RedisModuleString **argv, int argc);

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
