/**
 * dag.c
 *
 * Contains the helper methods for both parsing, running the command in the
 * background, and replying DAG structured commands.
 *
 * The way we allow DAG operations to run on different devices in parallel
 * (when possible) is the following: instead of running the whole DAG in one
 * swoop, the DAG run info is created on one
 * queue/device and shallow copied (appropriately) across other queues/devices
 * as indicated by the DAG specification. A DAG mutex is shared across all
 * copies.
 * The DAG run info is placed on the queue for each device and evicted for
 * execution (in background_workers). Execution happens one DAG op at a time:
 * once the individual op has executed, it is marked as such and the DAG run
 * info is placed back on the queue. The current un-executed op is checked for
 * its inputs. If all inputs are found in the tensor context, then the DAG op
 * can be executed. If not, the execution quits and control is given back to
 * the worker. If there are other items in the queue the op is placed after the
 * next item. When all ops for a device have been executed, the DAG is not
 * placed back on the queue. When all ops in a DAG have been executed or an
 * error occurs, the client is unblocked.
 *
 * See background_workers.c for the queue logic, everything else DAG is here.
 */

#include "dag.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "redisai.h"
#include "rmutil/args.h"
#include "rmutil/alloc.h"
#include "util/arr.h"
#include "util/dict.h"
#include "util/queue.h"
#include "util/string_utils.h"
#include "execution/run_info.h"
#include "execution/background_workers.h"
#include "execution/parsing/dag_parser.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/execution_contexts/scriptRun_ctx.h"
#include "execution/execution_contexts/execution_ctx.h"
#include "redis_ai_objects/model.h"
#include "redis_ai_objects/stats.h"
#include "redis_ai_objects/tensor.h"

static void Dag_LoadInputsToCurrentOp(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
    uint n_inkeys = array_len(currentOp->inkeys);
    uint n_outkeys = array_len(currentOp->outkeys);

    RAI_ContextReadLock(rinfo);

    RAI_Tensor *inputTensors[n_inkeys];
    for (uint i = 0; i < n_inkeys; i++) {
        RAI_Tensor *inputTensor = Dag_GetTensorFromGlobalCtx(rinfo, currentOp->inkeys_indices[i]);
        inputTensors[i] = inputTensor;
    }

    RAI_ContextUnlock(rinfo);

    // Input and output names should match to the one specified when the model was set, only in TF.
    // For other backends, model->inputs and model->outputs is null.
    for (uint i = 0; i < n_inkeys; i++) {
        const char *opname = NULL;
        RAI_ExecutionCtx_AddInput(currentOp->ectx, inputTensors[i]);
    }

    for (uint i = 0; i < n_outkeys; i++) {
        RAI_ExecutionCtx_AddOuputPlaceholder(currentOp->ectx);
    }
}

static void Dag_StoreCurrentOpOutputs(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {

    RAI_ContextWriteLock(rinfo);
    const size_t noutputs = RAI_ExecutionCtx_NumOutputs(currentOp->ectx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RAI_Tensor *tensor = RAI_ExecutionCtx_GetOutput(currentOp->ectx, outputNumber);
        Dag_SetTensorInGlobalCtx(rinfo, currentOp->outkeys_indices[outputNumber], tensor);
    }
    RAI_ContextUnlock(rinfo);
}

static int _StoreTensorInKeySpace(RedisModuleCtx *ctx, RAI_Tensor *tensor,
                                  RedisModuleString *persist_key_name, RAI_Error *err) {

    RedisModuleKey *key;
    int status =
        RAI_TensorOpenKey(ctx, persist_key_name, &key, REDISMODULE_READ | REDISMODULE_WRITE, err);
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeSetValue(key, RAI_TensorRedisType(), tensor) != REDISMODULE_OK) {
        RAI_SetError(err, RAI_EDAGRUN, "ERR could not save tensor");
        RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }
    // Only if we got until here, tensor is saved in keyspace.
    RAI_TensorReplicate(ctx, persist_key_name, tensor);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}

static int _DAG_PersistTensors(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo) {

    AI_dictIterator *persist_iter = AI_dictGetSafeIterator(rinfo->persistTensors);
    AI_dictEntry *persist_entry;

    while ((persist_entry = AI_dictNext(persist_iter))) {
        RedisModuleString *persist_key_name = AI_dictGetKey(persist_entry);
        size_t index = (size_t)AI_dictGetVal(persist_entry);
        RAI_Tensor *tensor = Dag_GetTensorFromGlobalCtx(rinfo, index);
        tensor = RAI_TensorGetShallowCopy(tensor);

        if (_StoreTensorInKeySpace(ctx, tensor, persist_key_name, rinfo->err) == REDISMODULE_ERR) {
            *rinfo->dagError = 1;
            RedisModule_Log(ctx, "warning",
                            "Could not persist tensor under the key (%s) after executing DAGRUN "
                            "command, persist stopped",
                            RedisModule_StringPtrLen(persist_key_name, NULL));
            AI_dictReleaseIterator(persist_iter);
            rinfo->dagReplyLength++;
            return REDISMODULE_ERR;
        }
    }
    AI_dictReleaseIterator(persist_iter);
    return REDISMODULE_OK;
}

static int _Dag_SingleOpPersistTensors(RedisModuleCtx *ctx, RAI_DagOp *op, RAI_Error *err) {

    const size_t noutputs = RAI_ExecutionCtx_NumOutputs(op->ectx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RedisModuleString *persist_key_name = op->outkeys[outputNumber];
        RAI_Tensor *tensor = RAI_ExecutionCtx_GetOutput(op->ectx, outputNumber);
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        if (!tensor)
            continue;

        if (_StoreTensorInKeySpace(ctx, tensor, persist_key_name, err) == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning",
                            "Could not persist tensor under the key (%s) after executing DAGRUN "
                            "command, persist stopped",
                            RedisModule_StringPtrLen(persist_key_name, NULL));
            op->result = REDISMODULE_ERR;
            return REDISMODULE_ERR;
        }
    }
    return REDISMODULE_OK;
}

/**
 * Execution of a MODELRUN DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp MODELRUN DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_ModelRun_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {

    // Get the needed tensors from the DAG local context. If the DAG originated
    // from a model run command, we are ready to execute.
    if (rinfo->single_op_dag == 0)
        Dag_LoadInputsToCurrentOp(rinfo, currentOp);

    RAI_ExecutionCtx *ectxs[1];
    ectxs[0] = currentOp->ectx;
    RAI_Model *model = RAI_ModelRunCtxGetModel((RAI_ModelRunCtx *)ectxs[0]);
    const long long start = ustime();
    int result = RAI_ModelRun((RAI_ModelRunCtx **)ectxs, 1, currentOp->err);
    const long long end = ustime();

    currentOp->duration_us = end - start;
    currentOp->result = result;
    if (result == REDISMODULE_ERR)
        return;
    if (rinfo->single_op_dag == 0)
        Dag_StoreCurrentOpOutputs(rinfo, currentOp);
}

/**
 * Execution of a batched (MODELRUN) DAG step.
 * If an error occurs, it is recorded in all DagOp structs.
 *
 * @param batched_rinfo array of contexts in which RedisAI blocking commands operate.
 * @param currentOps MODELRUN DagOps to be executed
 * @return
 */
void RedisAI_BatchedDagRunSession_ModelRun_Step(RedisAI_RunInfo **batched_rinfo,
                                                RAI_DagOp **currentOps) {

    int n_rinfo = array_len(batched_rinfo);
    RAI_ExecutionCtx *ectxs[n_rinfo];

    for (int i = 0; i < n_rinfo; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = currentOps[i];

        // Get the needed tensors from the DAG local context. If the DAG originated
        // from a model run command, we are ready to execute.
        if (rinfo->single_op_dag == 0)
            Dag_LoadInputsToCurrentOp(rinfo, currentOp);
        ectxs[i] = currentOp->ectx;
    }

    RAI_Error err = {0};
    const long long start = ustime();
    int result = RAI_ModelRun((RAI_ModelRunCtx **)ectxs, n_rinfo, &err);
    const long long end = ustime();

    long long duration = end - start;

    for (int i = 0; i < n_rinfo; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = currentOps[i];
        currentOp->duration_us = duration;
        currentOp->result = result;

        if (result == REDISMODULE_ERR) {
            RAI_SetError(currentOp->err, err.code, err.detail);
            continue;
        }
        if (rinfo->single_op_dag == 0)
            Dag_StoreCurrentOpOutputs(rinfo, currentOp);
    }
    // Clear the result in case of an error.
    if (result == REDISMODULE_ERR)
        RAI_ClearError(&err);
}

/**
 * Execution of a SCRIPTRUN DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp SCRIPTRUN DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_ScriptRun_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
    uint n_inkeys = array_len(currentOp->inkeys);
    uint n_outkeys = array_len(currentOp->outkeys);

    if (!rinfo->single_op_dag) {
        RAI_ContextReadLock(rinfo);
        RAI_Tensor *inputTensors[n_inkeys];
        for (uint i = 0; i < n_inkeys; i++) {
            RAI_Tensor *inputTensor =
                Dag_GetTensorFromGlobalCtx(rinfo, currentOp->inkeys_indices[i]);
            inputTensors[i] = inputTensor;
        }
        RAI_ContextUnlock(rinfo);
        for (uint i = 0; i < n_inkeys; i++) {
            RAI_ExecutionCtx_AddInput(currentOp->ectx, inputTensors[i]);
        }
        for (uint i = 0; i < n_outkeys; i++) {
            RAI_ExecutionCtx_AddOuputPlaceholder(currentOp->ectx);
        }
    }

    const long long start = ustime();
    int result = RAI_ScriptRun((RAI_ScriptRunCtx *)currentOp->ectx, currentOp->err);
    const long long end = ustime();

    currentOp->result = result;
    currentOp->duration_us = end - start;
    if (result != REDISMODULE_OK) {
        return;
    }

    if (!rinfo->single_op_dag) {
        RAI_ContextWriteLock(rinfo);
        const size_t noutputs = RAI_ExecutionCtx_NumOutputs(currentOp->ectx);
        for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
            RAI_Tensor *tensor = RAI_ExecutionCtx_GetOutput(currentOp->ectx, outputNumber);
            Dag_SetTensorInGlobalCtx(rinfo, currentOp->outkeys_indices[outputNumber], tensor);
        }
        RAI_ContextUnlock(rinfo);
    }
}

size_t RAI_DagOpBatchSize(RAI_DagOp *op, RedisAI_RunInfo *rinfo) {
    if (op->commandType != REDISAI_DAG_CMD_MODELRUN) {
        return -1;
    }

    size_t ninputs = array_len(op->inkeys);
    int batchsize = 0;

    RAI_ContextReadLock(rinfo);
    for (size_t i = 0; i < ninputs; i++) {
        RAI_Tensor *input;
        if (rinfo->single_op_dag) {
            input = RAI_ExecutionCtx_GetInput(op->ectx, i);
        } else {
            input = Dag_GetTensorFromGlobalCtx(rinfo, op->inkeys_indices[i]);
        }

        if (i == 0) {
            batchsize = RAI_TensorDim(input, 0);
            continue;
        }
        if (batchsize != RAI_TensorDim(input, 0)) {
            batchsize = 0;
            break;
        }
    }
    RAI_ContextUnlock(rinfo);

    return batchsize;
}

bool RAI_DagOpBatchable(RAI_DagOp *op1, RedisAI_RunInfo *rinfo1, RAI_DagOp *op2,
                        RedisAI_RunInfo *rinfo2) {

    RedisModule_Assert(op1->commandType == REDISAI_DAG_CMD_MODELRUN &&
                       op2->commandType == REDISAI_DAG_CMD_MODELRUN);
    RAI_ExecutionCtx *ectx1 = op1->ectx;
    RAI_ExecutionCtx *ectx2 = op2->ectx;
    if (RAI_ModelRunCtxGetModel((RAI_ModelRunCtx *)ectx1) !=
        RAI_ModelRunCtxGetModel((RAI_ModelRunCtx *)ectx2)) {
        return false;
    }
    const int ninputs1 = array_len(op1->inkeys);
    const int ninputs2 = array_len(op2->inkeys);
    if (ninputs1 != ninputs2) {
        return false;
    }

    RAI_ContextReadLock(rinfo1);
    RAI_ContextReadLock(rinfo2);
    for (int i = 0; i < ninputs1; i++) {
        RAI_Tensor *input1;
        if (rinfo1->single_op_dag) {
            input1 = RAI_ExecutionCtx_GetInput(ectx1, i);
        } else {
            input1 = Dag_GetTensorFromGlobalCtx(rinfo1, op1->inkeys_indices[i]);
        }
        RAI_Tensor *input2;
        if (rinfo2->single_op_dag) {
            input2 = RAI_ExecutionCtx_GetInput(ectx2, i);
        } else {
            input2 = Dag_GetTensorFromGlobalCtx(rinfo2, op2->inkeys_indices[i]);
        }
        int ndims1 = RAI_TensorNumDims(input1);
        int ndims2 = RAI_TensorNumDims(input2);
        if (ndims1 != ndims2) {
            return false;
        }
        if (ndims1 == 0) {
            continue;
        }
        for (int j = 1; j < ndims1; j++) {
            long long dim1 = RAI_TensorDim(input1, j);
            long long dim2 = RAI_TensorDim(input2, j);
            if (dim1 != dim2) {
                return false;
            }
        }
    }
    RAI_ContextUnlock(rinfo1);
    RAI_ContextUnlock(rinfo2);

    return true;
}

bool RedisAI_DagDeviceComplete(RedisAI_RunInfo *rinfo) {
    return rinfo->dagDeviceCompleteOpCount == rinfo->dagDeviceOpCount;
}

bool RedisAI_DagComplete(RedisAI_RunInfo *rinfo) {
    int completeOpCount = __atomic_load_n(rinfo->dagCompleteOpCount, __ATOMIC_RELAXED);

    return completeOpCount == rinfo->dagOpCount;
}

bool RedisAI_DagError(RedisAI_RunInfo *rinfo) {
    return __atomic_load_n(rinfo->dagError, __ATOMIC_RELAXED) != 0;
}

bool RedisAI_DagTimeout(RedisAI_RunInfo *rinfo) {
    __atomic_load_n(rinfo->timedOut, __ATOMIC_RELAXED) != 0;
}

void RedisAI_DagSetTimeout(RedisAI_RunInfo *rinfo) {
    __atomic_store_n(rinfo->timedOut, 1, __ATOMIC_RELAXED);
}

RAI_DagOp *RedisAI_DagCurrentOp(RedisAI_RunInfo *rinfo) {
    if (rinfo->dagDeviceCompleteOpCount == rinfo->dagDeviceOpCount) {
        return NULL;
    }

    return rinfo->dagDeviceOps[rinfo->dagDeviceCompleteOpCount];
}

void RedisAI_DagCurrentOpInfo(RedisAI_RunInfo *rinfo, bool *currentOpReady,
                              bool *currentOpBatchable) {

    RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);
    *currentOpReady = false;
    *currentOpBatchable = false;
    RedisModule_Assert(currentOp);
    if (currentOp->commandType == REDISAI_DAG_CMD_MODELRUN) {
        RAI_ModelRunCtx *mctx = (RAI_ModelRunCtx *)currentOp->ectx;
        RAI_Model *model = RAI_ModelRunCtxGetModel(mctx);
        // TODO: Remove abstraction break
        if (model->opts.batchsize > 0) {
            *currentOpBatchable = true;
        }
    }

    *currentOpReady = true;
    // If this is a single op dag, the op is definitely ready.
    if (rinfo->single_op_dag)
        return;

    uint n_inkeys = array_len(currentOp->inkeys);
    RAI_ContextReadLock(rinfo);

    for (int i = 0; i < n_inkeys; i++) {
        if (Dag_GetTensorFromGlobalCtx(rinfo, currentOp->inkeys_indices[i]) == NULL) {
            RAI_ContextUnlock(rinfo);
            *currentOpReady = false;
            return;
        }
    }
    RAI_ContextUnlock(rinfo);
}

void RedisAI_DagOpBatchInfo(RedisAI_RunInfo *rinfo, RAI_DagOp *op, size_t *batchsize,
                            size_t *minbatchsize, size_t *minbatchtimeout, size_t *inbatchsize) {
    *batchsize = 0;
    *minbatchsize = 0;
    *minbatchtimeout = 0;
    *inbatchsize = 0;
    if (op->commandType != REDISAI_DAG_CMD_MODELRUN)
        return;
    RAI_ModelRunCtx *mctx = (RAI_ModelRunCtx *)op->ectx;
    RAI_Model *model = RAI_ModelRunCtxGetModel(mctx);

    // TODO: clear abstraction breaks.
    *batchsize = model->opts.batchsize;
    *minbatchsize = model->opts.minbatchsize;
    *minbatchtimeout = model->opts.minbatchtimeout;
    *inbatchsize = RAI_DagOpBatchSize(op, rinfo);
}

void RedisAI_DagOpBatchingMatch(RedisAI_RunInfo *rinfo1, RAI_DagOp *op1, RedisAI_RunInfo *rinfo2,
                                RAI_DagOp *op2, int *batched, size_t *inbatchsize) {
    *batched = 0;
    *inbatchsize = 0;

    if (op2->commandType == REDISAI_DAG_CMD_MODELRUN) {
        bool match = RAI_DagOpBatchable(op1, rinfo1, op2, rinfo2);
        if (match) {
            *batched = 1;
            *inbatchsize = RAI_DagOpBatchSize(op2, rinfo2);
        }
    }
}

RAI_Tensor *Dag_GetTensorFromGlobalCtx(RedisAI_RunInfo *rinfo, size_t index) {
    RedisModule_Assert(index < array_len(rinfo->dagSharedTensors));
    return rinfo->dagSharedTensors[index];
}

void Dag_SetTensorInGlobalCtx(RedisAI_RunInfo *rinfo, size_t index, RAI_Tensor *t) {
    RedisModule_Assert(index < array_len(rinfo->dagSharedTensors));
    RedisModule_Assert(rinfo->dagSharedTensors[index] == NULL);
    rinfo->dagSharedTensors[index] = RAI_TensorGetShallowCopy(t);
}

void RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char *devicestr) {
    RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);

    // Verify that the op type belongs to the DAGCommand enum.
    VALIDATE_DAG_COMMAND(currentOp->commandType)

    if (currentOp->commandType == REDISAI_DAG_CMD_MODELRUN) {
        RedisAI_DagRunSession_ModelRun_Step(rinfo, currentOp);
    } else if (currentOp->commandType == REDISAI_DAG_CMD_SCRIPTRUN) {
        RedisAI_DagRunSession_ScriptRun_Step(rinfo, currentOp);
    } else {
        // do nothing for tensorset (executed on parsing) and for tensorget (done
        // on dag reply).
        return;
    }

    if (currentOp->result != REDISMODULE_OK) {
        // If this is the first op with error, save the error in the DAG runInfo.
        if (__sync_val_compare_and_swap(rinfo->dagError, 0, 1) == 0) {
            RAI_SetError(rinfo->err, RAI_GetErrorCode(currentOp->err),
                         RAI_GetError(currentOp->err));
        }
    }
}

void RedisAI_BatchedDagRunSessionStep(RedisAI_RunInfo **batched_rinfo, const char *devicestr) {
    // Assumption: ops are guaranteed to be all MODELRUN

    int n_ops = array_len(batched_rinfo);
    assert(n_ops > 1);
    RAI_DagOp *currentOps[n_ops];

    for (int i = 0; i < n_ops; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);
        currentOps[i] = currentOp;
    }

    RedisAI_BatchedDagRunSession_ModelRun_Step(batched_rinfo, currentOps);

    for (int i = 0; i < n_ops; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = currentOps[i];

        if (currentOp->result != REDISMODULE_OK) {
            // If this is the first op with error, save the error in the DAG runInfo.
            if (__sync_val_compare_and_swap(rinfo->dagError, 0, 1) == 0) {
                RAI_SetError(rinfo->err, RAI_GetErrorCode(currentOp->err),
                             RAI_GetError(currentOp->err));
            }
        }
    }
}

int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);
    RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

    if (*rinfo->timedOut) {
        RedisModule_ReplyWithSimpleString(ctx, "TIMEDOUT");
        return REDISMODULE_OK;
    }

    if (RAI_GetErrorCode(rinfo->err) == RAI_EDAGRUN) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
        return REDISMODULE_OK;
    }
    int dag_error = 0;
    size_t n_dagOps = array_len(rinfo->dagOps);

    if (!rinfo->single_op_dag) {
        RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
    }

    for (size_t i = 0; i < n_dagOps; i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        switch (currentOp->commandType) {
        case REDISAI_DAG_CMD_TENSORSET: {
            rinfo->dagReplyLength++;
            RedisModule_Assert(currentOp->result == REDISMODULE_OK);
            RedisModule_ReplyWithSimpleString(ctx, "OK");
            break;
        }

        case REDISAI_DAG_CMD_TENSORGET: {
            rinfo->dagReplyLength++;
            RAI_Tensor *t = Dag_GetTensorFromGlobalCtx(rinfo, currentOp->inkeys_indices[0]);
            if (t == NULL) {
                RedisModule_ReplyWithSimpleString(ctx, "NA");
            } else {
                RAI_TensorReply(ctx, currentOp->fmt, t);
            }
            break;
        }

        case REDISAI_DAG_CMD_MODELRUN: {
            rinfo->dagReplyLength++;
            struct RedisAI_RunStats *rstats = NULL;
            RAI_GetRunStats(currentOp->runkey, &rstats);
            if (currentOp->result == REDISMODULE_ERR) {
                RAI_SafeAddDataPoint(rstats, 0, 1, 1, 0);
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                dag_error = 1;
            } else if (currentOp->result == -1) {
                RedisModule_ReplyWithSimpleString(ctx, "NA");
            } else {
                RAI_Tensor *t = NULL;
                if (RAI_ExecutionCtx_NumOutputs(currentOp->ectx) > 0) {
                    t = RAI_ExecutionCtx_GetOutput(currentOp->ectx, 0);
                }
                int batch_size = 0;
                if (t) {
                    batch_size = RAI_TensorDim(t, 0);
                }
                RAI_SafeAddDataPoint(rstats, currentOp->duration_us, 1, 0, batch_size);
                RedisModule_ReplyWithSimpleString(ctx, "OK");
            }
            break;
        }

        case REDISAI_DAG_CMD_SCRIPTRUN: {
            rinfo->dagReplyLength++;
            struct RedisAI_RunStats *rstats = NULL;
            RAI_GetRunStats(currentOp->runkey, &rstats);
            if (currentOp->result == REDISMODULE_ERR) {
                RAI_SafeAddDataPoint(rstats, 0, 1, 1, 0);
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                dag_error = 1;
            } else if (currentOp->result == -1) {
                RedisModule_ReplyWithSimpleString(ctx, "NA");
            } else {
                int batch_size = 1;
                RAI_SafeAddDataPoint(rstats, currentOp->duration_us, 1, 0, batch_size);
                RedisModule_ReplyWithSimpleString(ctx, "OK");
            }
            break;
        }
        default:
            RedisModule_Assert(false && "Dag reply - invalid op");
        }
    }
    if (dag_error) {
        goto cleanup;
    }

    int persist_status;
    if (!rinfo->single_op_dag) {
        persist_status = _DAG_PersistTensors(ctx, rinfo);
    } else {
        persist_status = _Dag_SingleOpPersistTensors(ctx, rinfo->dagOps[0], rinfo->err);
    }
    if (persist_status != REDISMODULE_OK) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
    }

cleanup:
    if (!rinfo->single_op_dag) {
        RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
    }
    return REDISMODULE_OK;
}

int RedisAI_DagExecute_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                        RedisModuleString **argv, int argc) {

    size_t argpos = 1;
    while (argpos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos++], NULL);
        if (!strcasecmp(arg_string, "LOAD") || !strcasecmp(arg_string, "PERSIST")) {
            if (argpos >= argc) {
                return REDISMODULE_ERR;
            }
            long long n_keys;
            const int retval = RedisModule_StringToLongLong(argv[argpos++], &n_keys);
            if (retval != REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            size_t last_argpos = n_keys + argpos;
            if (last_argpos >= argc) {
                return REDISMODULE_ERR;
            }
            for (; argpos < last_argpos; argpos++) {
                RedisModule_KeyAtPos(ctx, argpos);
            }
        } else if (!strcasecmp(arg_string, "ROUTING")) {
            if (argpos >= argc) {
                return REDISMODULE_ERR;
            }
            RedisModule_KeyAtPos(ctx, argpos++);
        } else if (!strcasecmp(arg_string, "AI.MODELEXECUTE") ||
                   !strcasecmp(arg_string, "AI.SCRIPTEXECUTE")) {
            if (argpos >= argc) {
                return REDISMODULE_ERR;
            }
            // After every AI.MODEL/SCRIPTEXECUTE arg comes the model/script key.
            RedisModule_KeyAtPos(ctx, argpos++);
        }
    }
    return REDISMODULE_OK;
}

void RunInfo_FreeData(RedisModuleCtx *ctx, void *rinfo) { RAI_FreeRunInfo(rinfo); }

void DAG_ReplyAndUnblock(RedisAI_OnFinishCtx *ctx, void *private_data) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)ctx;
    if (rinfo->client) {
        if (RedisModule_GetServerVersion() >=
            0x060200) { // The following command is supported only from redis 6.2
            RedisModule_BlockedClientMeasureTimeEnd(rinfo->client);
        }
        RedisModule_UnblockClient(rinfo->client, rinfo);
    }
}
