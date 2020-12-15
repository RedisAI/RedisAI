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
 * Get whether all DAG ops for the given device have been executed
 * successfully. Since rinfo carries information on what queue
 * it has been placed in, there's no need to pass the device identifier.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @return nonzero if all ops are complete for device, 0 otherwise
 */
int RedisAI_DagDeviceComplete(RedisAI_RunInfo *rinfo);

/**
 * Get whether all DAG ops have been executed successfully irrespective
 * of the device, i.e. if the DAG has been completely executed.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @return nonzero of all ops in DAG are complete, 0 otherwise
 */
int RedisAI_DagComplete(RedisAI_RunInfo *rinfo);

/**
 * Get current DAG op for the given device. An op is current if it's
 * the first unrealized op for the device. Since rinfo carries information
 * on what queue it has been placed in, there's no need to pass the device
 * identifier.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @return pointer to current DAG op for device
 */
RAI_DagOp *RedisAI_DagCurrentOp(RedisAI_RunInfo *rinfo);

/**
 * Get information about current DAG op for the given device.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOpReady have all inputs for the op been computed, that is,
 *            are they available in the tensor context
 * @param currentOpBatchable is the op amenable to batching, that is, is it
 *            a MODELRUN and is BATCHSIZE greater than zero
 * @return
 */
void RedisAI_DagCurrentOpInfo(RedisAI_RunInfo *rinfo, int *currentOpReady, int *currentOpBatchable);

/**
 * Get batching information about a DAG op.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param op DAG operation
 * @param batchsize maximum batch size specified by BATCHSIZE
 * @param minbatchsize minimum batch size specified by MINBATCHSIZE
 * @param minbatchtimeout minimum batch timeout specified by MINBATCHTIMEOUT
 * @param inbatchsize actual size of the batch in the current op, that
 *            is, the size of the input tensors along the zero-th dimension
 * @return
 */
void RedisAI_DagOpBatchInfo(RedisAI_RunInfo *rinfo, RAI_DagOp *op, size_t *batchsize,
                            size_t *minbatchsize, size_t *minbatchtimeout, size_t *inbatchsize);

/**
 * Check that a DAG operation can be batched with a given batch operation.
 * @param rinfo1 given context in which RedisAI blocking commands operate.
 * @param op1 given DAG operation
 * @param rinfo2 other context in which RedisAI blocking commands operate.
 * @param op2 other DAG operation to be checked
 * @param batched can op2 be batched with op1
 * @param inbatchsize actual size of the batch in op2
 * @return
 */
void RedisAI_DagOpBatchingMatch(RedisAI_RunInfo *rinfo1, RAI_DagOp *op1, RedisAI_RunInfo *rinfo2,
                                RAI_DagOp *op2, int *batched, size_t *inbatchsize);

/**
 * Run the first unrealized DAG operation in rinfo for the given device.
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @return
 */
void RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char *devicestr);

/**
 * Batch the first unrealized DAG operations for the given device for the
 * provided rinfo and run.
 * @param rinfo contexts in which RedisAI blocking commands operate.
 * @param devicestr device identifier associated with the current queue
 * @return
 */
void RedisAI_BatchedDagRunSessionStep(RedisAI_RunInfo **rinfo, const char *devicestr);

/**
 * Reply Callback called after a successful RedisModule_UnblockClient() after
 * RedisAI_DagRunSession() in order to reply to the client and unblock it
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

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
int RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx, RedisModuleString **argv,
                                                    int argc);

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
int RedisAI_ProcessDagRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                 int dagMode);

/**
 * @brief This callback is called at the end of a DAG run and performs unblock client and reply.
 * This is the callback of RedisAI AI.MODELRUN, AI.SCRIPTRUN, AI.DAGRUN
 * @param ctx Context object that contains errors and results
 * @param private_data is a pointer to the DAG run info struct
 */
void DAG_ReplyAndUnblock(RedisAI_OnFinishCtx *ctx, void *private_data);

/**
 * @brief Insert DAG runInfo to the worker queues
 * @param RunInfo object to insert.
 */
int DAG_InsertDAGToQueue(RedisAI_RunInfo *rinfo);

/**
 * @brief A callback to send to BlockClient (we only send this function but we
 * don't use it for freeing the runInfo object, we use RAI_FreeRunInfo)
 */
void RunInfo_FreeData(RedisModuleCtx *ctx, void *rinfo);

/**
 * @brief A callback to send to BlockClient.
 */
void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc);

/**
 * @brief Populate a DAG with a single operation of MODELRUN.
 * @param rinfo An existing DAG to populate.
 * @param mctx ModelRunCtx that represents the single MODELRUN op.
 * @param inkeys The DAG operation inkeys (the input tensors).
 * @param outkeys The DAG operation outkeys (the output tensors).
 * @param runkey The model key.
 * @param timeout The operation timeout (if given, otherwise it is zero).
 */

int Dag_PopulateSingleModelRunOp(RedisAI_RunInfo *rinfo, RAI_ModelRunCtx *mctx,
                                 RedisModuleString **inkeys, RedisModuleString **outkeys,
                                 RedisModuleString *runkey, long long timeout);

#endif /* SRC_DAG_H_ */
