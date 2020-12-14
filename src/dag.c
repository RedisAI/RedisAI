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

#include "model.h"
#include "redisai.h"
#include "background_workers.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

/**
 * Execution of a TENSORSET DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp TENSORSET DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_TensorSet_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
    RAI_Tensor *t = NULL;
    const int parse_result =
        RAI_parseTensorSetArgs(NULL, currentOp->argv, currentOp->argc, &t, 0, currentOp->err);
    if (parse_result > 0) {
        const char *key_string = RedisModule_StringPtrLen(currentOp->outkeys[0], NULL);
        RAI_ContextWriteLock(rinfo);
        AI_dictReplace(rinfo->dagTensorsContext, (void *)key_string, t);
        RAI_ContextUnlock(rinfo);
        currentOp->result = REDISMODULE_OK;
    } else {
        currentOp->result = REDISMODULE_ERR;
    }
}

/**
 * Execution of a TENSORGET DAG step.
 * If an error occurs, it is recorded in the DagOp struct.
 *
 * @param rinfo context in which RedisAI blocking commands operate.
 * @param currentOp TENSORGET DagOp to be executed
 * @return
 */
void RedisAI_DagRunSession_TensorGet_Step(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
    const char *key_string = RedisModule_StringPtrLen(currentOp->inkeys[0], NULL);
    RAI_Tensor *t = NULL;
    RAI_ContextReadLock(rinfo);
    currentOp->result = RAI_getTensorFromLocalContext(NULL, rinfo->dagTensorsContext, key_string,
                                                      &t, currentOp->err);
    RAI_ContextUnlock(rinfo);
    if (currentOp->result == REDISMODULE_OK) {
        RAI_Tensor *outTensor = NULL;
        // TODO: check tensor copy return value
        RAI_TensorDeepCopy(t, &outTensor);
        currentOp->outTensors = array_append(currentOp->outTensors, outTensor);
    }
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
    uint n_inkeys = array_len(currentOp->inkeys);
    uint n_outkeys = array_len(currentOp->outkeys);

    RAI_ContextReadLock(rinfo);

    RAI_Tensor *inputTensors[n_inkeys];
    for (uint i = 0; i < n_inkeys; i++) {
        RAI_Tensor *inputTensor;
        const int get_result = RAI_getTensorFromLocalContext(
            NULL, rinfo->dagTensorsContext, RedisModule_StringPtrLen(currentOp->inkeys[i], NULL),
            &inputTensor, currentOp->err);
        if (get_result == REDISMODULE_ERR) {
            // We check for this outside the function
            // this check cannot be covered by tests
            currentOp->result = REDISMODULE_ERR;
            RAI_ContextUnlock(rinfo);
            return;
        }
        inputTensors[i] = inputTensor;
    }

    RAI_ContextUnlock(rinfo);

    for (uint i = 0; i < n_inkeys; i++) {
        const char *opname = NULL;
        if (currentOp->mctx->model->inputs) {
            opname = currentOp->mctx->model->inputs[i];
        }
        RAI_ModelRunCtxAddInput(currentOp->mctx, opname, inputTensors[i]);
    }

    for (uint i = 0; i < n_outkeys; i++) {
        const char *opname = NULL;
        if (currentOp->mctx->model->inputs) {
            opname = currentOp->mctx->model->outputs[i];
        }
        RAI_ModelRunCtxAddOutput(currentOp->mctx, opname);
    }

    RAI_ModelRunCtx *mctxs[1];
    mctxs[0] = currentOp->mctx;
    const long long start = ustime();
    int result = RAI_ModelRun(mctxs, 1, currentOp->err);
    const long long end = ustime();

    if (result == REDISMODULE_ERR) {
        currentOp->result = result;
        return;
    }

    RAI_ContextWriteLock(rinfo);

    currentOp->duration_us = end - start;

    const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RAI_Tensor *tensor = RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
        const char *key_string = RedisModule_StringPtrLen(currentOp->outkeys[outputNumber], NULL);
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        AI_dictReplace(rinfo->dagTensorsContext, (void *)key_string, tensor);
    }

    currentOp->result = result;

    RAI_ContextUnlock(rinfo);

    return;
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

    RAI_ModelRunCtx *mctxs[n_rinfo];

    for (int i = 0; i < n_rinfo; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = currentOps[i];

        uint n_inkeys = array_len(currentOp->inkeys);
        uint n_outkeys = array_len(currentOp->outkeys);

        RAI_ContextReadLock(rinfo);

        RAI_Tensor *inputTensors[n_inkeys];
        for (uint i = 0; i < n_inkeys; i++) {
            RAI_Tensor *inputTensor;
            const int get_result = RAI_getTensorFromLocalContext(
                NULL, rinfo->dagTensorsContext,
                RedisModule_StringPtrLen(currentOp->inkeys[i], NULL), &inputTensor, currentOp->err);
            if (get_result == REDISMODULE_ERR) {
                // We check for this outside the function
                // this check cannot be covered by tests
                currentOp->result = REDISMODULE_ERR;
                RAI_ContextUnlock(rinfo);
                return;
            }
            inputTensors[i] = inputTensor;
        }

        RAI_ContextUnlock(rinfo);

        for (uint i = 0; i < n_inkeys; i++) {
            const char *input_name = NULL;
            if (currentOp->mctx->model->inputs) {
                input_name = currentOp->mctx->model->inputs[i];
            }
            RAI_ModelRunCtxAddInput(currentOp->mctx, input_name, inputTensors[i]);
        }

        for (uint i = 0; i < n_outkeys; i++) {
            const char *output_name = NULL;
            if (currentOp->mctx->model->outputs) {
                output_name = currentOp->mctx->model->outputs[i];
            }
            RAI_ModelRunCtxAddOutput(currentOp->mctx, output_name);
        }

        mctxs[i] = currentOp->mctx;
    }

    RAI_Error err = {0};
    const long long start = ustime();
    int result = RAI_ModelRun(mctxs, n_rinfo, &err);
    const long long end = ustime();

    long long duration = end - start;

    for (int i = 0; i < n_rinfo; i++) {
        RedisAI_RunInfo *rinfo = batched_rinfo[i];
        RAI_DagOp *currentOp = currentOps[i];

        if (result == REDISMODULE_ERR) {
            currentOp->result = result;
            RAI_SetError(currentOp->err, err.code, err.detail);
            continue;
        }

        RAI_ContextWriteLock(rinfo);

        currentOp->duration_us = duration;

        const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
        for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
            RAI_Tensor *tensor = RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
            const char *key_string =
                RedisModule_StringPtrLen(currentOp->outkeys[outputNumber], NULL);
            tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
            AI_dictReplace(rinfo->dagTensorsContext, (void *)key_string, tensor);
        }

        currentOp->result = result;

        RAI_ContextUnlock(rinfo);
    }

    if (result == REDISMODULE_ERR) {
        RAI_ClearError(&err);
    }

    return;
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

    RAI_ContextReadLock(rinfo);

    RAI_Tensor *inputTensors[n_inkeys];
    for (uint i = 0; i < n_inkeys; i++) {
        RAI_Tensor *inputTensor;
        const int get_result = RAI_getTensorFromLocalContext(
            NULL, rinfo->dagTensorsContext, RedisModule_StringPtrLen(currentOp->inkeys[i], NULL),
            &inputTensor, currentOp->err);
        if (get_result == REDISMODULE_ERR) {
            // We check for this outside the function
            // this check cannot be covered by tests
            currentOp->result = REDISMODULE_ERR;
            RAI_ContextUnlock(rinfo);
            return;
        }
        inputTensors[i] = inputTensor;
    }

    RAI_ContextUnlock(rinfo);

    for (uint i = 0; i < n_inkeys; i++) {
        RAI_ScriptRunCtxAddInput(currentOp->sctx, inputTensors[i], currentOp->err);
    }

    for (uint i = 0; i < n_outkeys; i++) {
        RAI_ScriptRunCtxAddOutput(currentOp->sctx);
    }

    const long long start = ustime();
    int result = RAI_ScriptRun(currentOp->sctx, currentOp->err);
    const long long end = ustime();

    RAI_ContextWriteLock(rinfo);

    const size_t noutputs = RAI_ScriptRunCtxNumOutputs(currentOp->sctx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RAI_Tensor *tensor = RAI_ScriptRunCtxOutputTensor(currentOp->sctx, outputNumber);
        const char *key_string = RedisModule_StringPtrLen(currentOp->outkeys[outputNumber], NULL);
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        AI_dictReplace(rinfo->dagTensorsContext, (void *)key_string, tensor);
    }

    currentOp->result = result;
    currentOp->duration_us = end - start;

    RAI_ContextUnlock(rinfo);

    return;
}

size_t RAI_DagOpBatchSize(RAI_DagOp *op, AI_dict *opTensorsContext) {
    if (op->mctx == NULL) {
        return -1;
    }

    // size_t ninputs = RAI_ModelRunCtxNumInputs(op->mctx);
    size_t ninputs = array_len(op->inkeys);

    int batchsize = 0;

    if (ninputs == 0) {
        return batchsize;
    }

    for (size_t i = 0; i < ninputs; i++) {
        RAI_Tensor *input;
        RAI_getTensorFromLocalContext(
            NULL, opTensorsContext, RedisModule_StringPtrLen(op->inkeys[i], NULL), &input, op->err);
        // We are expecting input != NULL, because we only reach this function if all inputs
        // are available in context for the current dagOp. We could be more defensive eventually.

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

int RAI_DagOpBatchable(RAI_DagOp *op1, AI_dict *op1TensorsContext, RAI_DagOp *op2,
                       AI_dict *op2TensorsContext) {

    if (op1->mctx == NULL || op2->mctx == NULL) {
        return 0;
    }

    if (op1->mctx->model != op2->mctx->model) {
        return 0;
    }

    const int ninputs1 = RAI_ModelRunCtxNumInputs(op1->mctx);
    const int ninputs2 = RAI_ModelRunCtxNumInputs(op2->mctx);

    if (ninputs1 != ninputs2) {
        return 0;
    }

    for (int i = 0; i < ninputs1; i++) {
        RAI_Tensor *input1;
        RAI_getTensorFromLocalContext(NULL, op1TensorsContext,
                                      RedisModule_StringPtrLen(op1->inkeys[i], NULL), &input1,
                                      op1->err);

        RAI_Tensor *input2;
        RAI_getTensorFromLocalContext(NULL, op2TensorsContext,
                                      RedisModule_StringPtrLen(op2->inkeys[i], NULL), &input2,
                                      op2->err);

        if (input1 == NULL || input2 == NULL) {
            return 0;
        }

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

int RedisAI_DagDeviceComplete(RedisAI_RunInfo *rinfo) {
    return rinfo->dagDeviceCompleteOpCount == rinfo->dagDeviceOpCount;
}

int RedisAI_DagComplete(RedisAI_RunInfo *rinfo) {
    int completeOpCount = __atomic_load_n(rinfo->dagCompleteOpCount, __ATOMIC_RELAXED);

    return completeOpCount == rinfo->dagOpCount;
}

RAI_DagOp *RedisAI_DagCurrentOp(RedisAI_RunInfo *rinfo) {
    if (rinfo->dagDeviceCompleteOpCount == rinfo->dagDeviceOpCount) {
        return NULL;
    }

    return rinfo->dagDeviceOps[rinfo->dagDeviceCompleteOpCount];
}

void RedisAI_DagCurrentOpInfo(RedisAI_RunInfo *rinfo, int *currentOpReady,
                              int *currentOpBatchable) {
    RAI_DagOp *currentOp_ = RedisAI_DagCurrentOp(rinfo);

    *currentOpReady = 0;
    *currentOpBatchable = 0;

    if (currentOp_ == NULL) {
        return;
    }

    if (currentOp_->mctx && currentOp_->mctx->model->opts.batchsize > 0) {
        *currentOpBatchable = 1;
    }

    uint n_inkeys = array_len(currentOp_->inkeys);

    RAI_ContextReadLock(rinfo);

    *currentOpReady = 1;
    for (int i = 0; i < n_inkeys; i++) {
        if (AI_dictFind(rinfo->dagTensorsContext,
                        RedisModule_StringPtrLen(currentOp_->inkeys[i], NULL)) == NULL) {
            RAI_ContextUnlock(rinfo);
            *currentOpReady = 0;
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

    RAI_ContextReadLock(rinfo);

    if (op->mctx) {
        *batchsize = op->mctx->model->opts.batchsize;
        *minbatchsize = op->mctx->model->opts.minbatchsize;
        *minbatchtimeout = op->mctx->model->opts.minbatchtimeout;
        *inbatchsize = RAI_DagOpBatchSize(op, rinfo->dagTensorsContext);
    }

    RAI_ContextUnlock(rinfo);
}

void RedisAI_DagOpBatchingMatch(RedisAI_RunInfo *rinfo1, RAI_DagOp *op1, RedisAI_RunInfo *rinfo2,
                                RAI_DagOp *op2, int *batched, size_t *inbatchsize) {
    *batched = 0;
    *inbatchsize = 0;

    RAI_ContextReadLock(rinfo2);

    if (op2->mctx) {
        int match =
            RAI_DagOpBatchable(op1, rinfo1->dagTensorsContext, op2, rinfo2->dagTensorsContext);

        if (match) {
            *batched = 1;
            *inbatchsize = RAI_DagOpBatchSize(op2, rinfo2->dagTensorsContext);
        }
    }

    RAI_ContextUnlock(rinfo2);
}

void RedisAI_DagRunSessionStep(RedisAI_RunInfo *rinfo, const char *devicestr) {
    RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);

    switch (currentOp->commandType) {
    case REDISAI_DAG_CMD_TENSORSET: {
        RedisAI_DagRunSession_TensorSet_Step(rinfo, currentOp);
        break;
    }
    case REDISAI_DAG_CMD_TENSORGET: {
        RedisAI_DagRunSession_TensorGet_Step(rinfo, currentOp);
        break;
    }
    case REDISAI_DAG_CMD_MODELRUN: {
        RedisAI_DagRunSession_ModelRun_Step(rinfo, currentOp);
        break;
    }
    case REDISAI_DAG_CMD_SCRIPTRUN: {
        RedisAI_DagRunSession_ScriptRun_Step(rinfo, currentOp);
        break;
    }
    default: {
        /* unsupported DAG's command */
        RAI_SetError(currentOp->err, RAI_EDAGRUN, "ERR unsupported command within DAG");
        currentOp->result = REDISMODULE_ERR;
        break;
    }
    }

    if (currentOp->result != REDISMODULE_OK) {
        __atomic_store_n(rinfo->dagError, 1, __ATOMIC_RELAXED);
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
            __atomic_store_n(rinfo->dagError, 1, __ATOMIC_RELAXED);
        }
    }

    return;
}

int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);
    RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

    if (RAI_GetErrorCode(rinfo->err) == RAI_EDAGRUN) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_ERR;
    }
    int dag_error = 0;
    char *detail_oneline;

    size_t n_dagOps = array_len(rinfo->dagOps);

    if (*rinfo->timedOut) {
        RedisModule_ReplyWithSimpleString(ctx, "TIMEDOUT");
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_OK;
    }

    if (rinfo->single_op_dag == 0) {
        RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
    }

    for (size_t i = 0; i < n_dagOps; i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];
        switch (currentOp->commandType) {
        case REDISAI_DAG_CMD_TENSORSET: {
            rinfo->dagReplyLength++;
            if (currentOp->result == REDISMODULE_ERR) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                dag_error = 1;
            } else if (currentOp->result == -1) {
                RedisModule_ReplyWithSimpleString(ctx, "NA");
            } else {
                RedisModule_ReplyWithSimpleString(ctx, "OK");
            }
            break;
        }

        case REDISAI_DAG_CMD_TENSORGET: {
            rinfo->dagReplyLength++;
            if (currentOp->result == REDISMODULE_ERR) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                dag_error = 1;
            } else {
                if (array_len(currentOp->outTensors) > 0) {
                    RAI_Tensor *tensor = currentOp->outTensors[0];
                    RAI_parseTensorGetArgs(ctx, currentOp->argv, currentOp->argc, tensor);
                } else if (currentOp->result == -1) {
                    RedisModule_ReplyWithSimpleString(ctx, "NA");
                } else {
                    RedisModule_ReplyWithError(ctx, "ERR error getting tensor from local context");
                }
            }
            break;
        }

        case REDISAI_DAG_CMD_MODELRUN: {
            rinfo->dagReplyLength++;
            struct RedisAI_RunStats *rstats = NULL;
            const char *runkey = RedisModule_StringPtrLen(currentOp->runkey, NULL);
            RAI_GetRunStats(runkey, &rstats);
            if (currentOp->result == REDISMODULE_ERR) {
                RAI_SafeAddDataPoint(rstats, 0, 1, 1, 0);
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                dag_error = 1;
            } else if (currentOp->result == -1) {
                RedisModule_ReplyWithSimpleString(ctx, "NA");
            } else {
                RAI_Tensor *t = NULL;
                if (array_len(currentOp->mctx->outputs) > 0) {
                    t = currentOp->mctx->outputs[0].tensor;
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
            const char *runkey = RedisModule_StringPtrLen(currentOp->runkey, NULL);
            RAI_GetRunStats(runkey, &rstats);
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
            /* no-op */
            break;
        }
    }

    if (dag_error) {
        if (rinfo->single_op_dag == 0) {
            RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
        }
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_ERR;
    }

    AI_dictIterator *persist_iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
    AI_dictEntry *persist_entry = AI_dictNext(persist_iter);
    while (persist_entry) {
        const char *persist_key_name = AI_dictGetKey(persist_entry);

        AI_dictEntry *tensor_entry = AI_dictFind(rinfo->dagTensorsContext, persist_key_name);

        if (tensor_entry) {
            RAI_Tensor *tensor = AI_dictGetVal(tensor_entry);

            if (tensor == NULL) {
                persist_entry = AI_dictNext(persist_iter);
                continue;
            }
            RedisModuleKey *key;
            char *demangled_key_name = RedisModule_Strdup(persist_key_name);
            demangled_key_name[strlen(persist_key_name) - 4] = 0;
            RedisModuleString *tensor_keyname =
                RedisModule_CreateString(ctx, demangled_key_name, strlen(demangled_key_name));
            const int status =
                RAI_OpenKey_Tensor(ctx, tensor_keyname, &key, REDISMODULE_READ | REDISMODULE_WRITE);
            RedisModule_Free(demangled_key_name);
            if (status == REDISMODULE_ERR) {
                RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
                rinfo->dagReplyLength++;
            } else {
                if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType,
                                                   RAI_TensorGetShallowCopy(tensor)) !=
                    REDISMODULE_OK) {
                    RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
                    rinfo->dagReplyLength++;
                }
            }
            RedisModule_CloseKey(key);
            RedisAI_ReplicateTensorSet(ctx, tensor_keyname, tensor);
        } else {
            RedisModule_ReplyWithError(ctx,
                                       "ERR specified persistent key that was not used in DAG");
            rinfo->dagReplyLength++;

            RedisModule_Log(ctx, "warning",
                            "on DAGRUN's PERSIST pecified persistent key (%s) that "
                            "was not used on DAG. Logging all local context keys",
                            persist_key_name);
            AI_dictIterator *local_iter = AI_dictGetSafeIterator(rinfo->dagTensorsContext);
            AI_dictEntry *local_entry = AI_dictNext(local_iter);
            while (local_entry) {
                const char *localcontext_key_name = AI_dictGetKey(local_entry);
                RedisModule_Log(ctx, "warning", "DAG's local context key (%s)",
                                localcontext_key_name);
                local_entry = AI_dictNext(local_iter);
            }
            AI_dictReleaseIterator(local_iter);

            for (size_t opN = 0; opN < array_len(rinfo->dagOps); opN++) {
                RedisModule_Log(ctx, "warning", "DAG's op n#  %zu - cmdType %d ( argc %d )", opN,
                                rinfo->dagOps[opN]->commandType, rinfo->dagOps[opN]->argc);
            }
        }

        persist_entry = AI_dictNext(persist_iter);
    }

    AI_dictReleaseIterator(persist_iter);

    if (rinfo->single_op_dag == 0) {
        RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
    }

    RAI_FreeRunInfo(rinfo);

    return REDISMODULE_OK;
}

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                         AI_dict **loadedContextDict, AI_dict **localContextDict,
                         const char *chaining_operator) {
    if (argc < 3) {
        RedisModule_WrongArity(ctx);
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RedisModule_ReplyWithError(ctx,
                                   "ERR invalid or negative value found in number of keys to LOAD");
        return -1;
    }
    int number_loaded_keys = 0;
    int separator_flag = 0;
    size_t argpos = 2;
    for (; (argpos <= argc - 1) && (number_loaded_keys < n_keys); argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, chaining_operator)) {
            separator_flag = 1;
            break;
        } else {
            RAI_Tensor *t;
            RedisModuleKey *key;
            const int status =
                RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t, REDISMODULE_READ);
            if (status == REDISMODULE_ERR) {
                RedisModule_Log(ctx, "warning",
                                "on DAGRUN's LOAD could not load tensor %s from keyspace",
                                arg_string);
                return -1;
            }
            RedisModule_CloseKey(key);
            char *dictKey = (char *)RedisModule_Alloc((strlen(arg_string) + 5) * sizeof(char));
            sprintf(dictKey, "%s%04d", arg_string, 1);
            AI_dictAdd(*localContextDict, (void *)dictKey, (void *)RAI_TensorGetShallowCopy(t));
            AI_dictAdd(*loadedContextDict, (void *)dictKey, (void *)1);
            RedisModule_Free(dictKey);
            number_loaded_keys++;
        }
    }
    if (number_loaded_keys != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return argpos;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                            AI_dict **persistContextDict, const char *chaining_operator) {
    if (argc < 3) {
        RedisModule_WrongArity(ctx);
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RedisModule_ReplyWithError(
            ctx, "ERR invalid or negative value found in number of keys to PERSIST");
        return -1;
    }

    int number_loaded_keys = 0;
    int separator_flag = 0;
    size_t argpos = 2;
    for (; (argpos < argc) && (number_loaded_keys < n_keys); argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, chaining_operator)) {
            separator_flag = 1;
            break;
        } else {
            AI_dictAdd(*persistContextDict, (void *)arg_string, (void *)1);
            number_loaded_keys++;
        }
    }
    if (number_loaded_keys != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return argpos;
}

int RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx, RedisModuleString **argv,
                                                    int argc) {
    for (size_t argpos = 1; argpos < argc; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if ((!strcasecmp(arg_string, "LOAD") || !strcasecmp(arg_string, "PERSIST")) &&
            (argpos + 1 < argc)) {
            long long n_keys;
            argpos++;
            const int retval = RedisModule_StringToLongLong(argv[argpos], &n_keys);
            if (retval != REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            argpos++;
            if (n_keys > 0) {
                size_t last_persist_argpos = n_keys + argpos;
                for (; argpos < last_persist_argpos && argpos < argc; argpos++) {
                    RedisModule_KeyAtPos(ctx, argpos);
                }
            }
        }
    }
    return REDISMODULE_OK;
}

void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo) {}

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
    RedisModule_Log(ctx, "warning", "Blocked client %p disconnected!", (void *)bc);
}

// Parse the DAG run command and return REDISMODULE_OK only if it is a valid command to execute.
static int DAG_CommandParser(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, int dagMode,
                             RedisAI_RunInfo **rinfo_ptr) {

    if (argc < 4) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }
    RedisAI_RunInfo *rinfo = *rinfo_ptr;
    RAI_DagOp *currentDagOp = NULL;
    RAI_InitDagOp(&currentDagOp);
    rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);

    int persistFlag = 0;
    int loadFlag = 0;
    int chainingOpCount = 0;

    int argstart = 1;

    // If we're parsing a AI.MODELRUN or AI.SCRIPTRUN command, we don't
    // expect there to be a chaining |> operator
    if (!strcasecmp(RedisModule_StringPtrLen(argv[0], NULL), "AI.MODELRUN") ||
        !strcasecmp(RedisModule_StringPtrLen(argv[0], NULL), "AI.SCRIPTRUN")) {
        argstart = 0;
        chainingOpCount++;
        rinfo->single_op_dag = 1;
        rinfo->single_device_dag = 1;
    }

    for (size_t argpos = argstart; argpos <= argc - 1; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "LOAD")) {
            loadFlag = 1;
            const int parse_result = RAI_parseDAGLoadArgs(ctx, &argv[argpos], argc - argpos,
                                                          &(rinfo->dagTensorsLoadedContext),
                                                          &(rinfo->dagTensorsContext), "|>");
            if (parse_result > 0) {
                argpos += parse_result - 1;
            } else {
                RAI_FreeRunInfo(rinfo);
                return REDISMODULE_ERR;
            }
        } else if (!strcasecmp(arg_string, "PERSIST")) {
            if (dagMode == REDISAI_DAG_READONLY_MODE) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx,
                                           "ERR PERSIST cannot be specified in a read-only DAG");
                return REDISMODULE_ERR;
            }
            persistFlag = 1;
            const int parse_result = RAI_parseDAGPersistArgs(
                ctx, &argv[argpos], argc - argpos, &(rinfo->dagTensorsPersistedContext), "|>");
            if (parse_result > 0) {
                argpos += parse_result - 1;
            } else {
                RAI_FreeRunInfo(rinfo);
                return REDISMODULE_ERR;
            }
        } else if (!strcasecmp(arg_string, "TIMEOUT")) {
            if (!((chainingOpCount == 0) || (chainingOpCount == 1 && rinfo->single_op_dag == 1))) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR TIMEOUT not allowed within a DAG command");
                return REDISMODULE_ERR;
            }
            if (argpos == argc - 1) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR No value provided for TIMEOUT");
                return REDISMODULE_ERR;
            }
            long long timeout;
            const int retval = RedisModule_StringToLongLong(argv[argpos + 1], &timeout);
            if (retval != REDISMODULE_OK || timeout <= 0) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR Invalid value for TIMEOUT");
                return REDISMODULE_ERR;
            }
            rinfo->timeout = timeout;
            argpos += 1;
            continue;
        } else if (!strcasecmp(arg_string, "|>") && argpos < argc - 1) {
            // on the first pipe operator, if LOAD or PERSIST were used, we've already
            // allocated memory
            if (chainingOpCount > 0) {
                rinfo->dagOpCount++;
                RAI_DagOp *currentDagOp = NULL;
                RAI_InitDagOp(&currentDagOp);
                rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
            }
            chainingOpCount++;
        } else {
            if (!strcasecmp(arg_string, "AI.TENSORGET")) {
                rinfo->dagOps[rinfo->dagOpCount]->commandType = REDISAI_DAG_CMD_TENSORGET;
                rinfo->dagOps[rinfo->dagOpCount]->devicestr = "CPU";
            }
            if (!strcasecmp(arg_string, "AI.TENSORSET")) {
                rinfo->dagOps[rinfo->dagOpCount]->commandType = REDISAI_DAG_CMD_TENSORSET;
                rinfo->dagOps[rinfo->dagOpCount]->devicestr = "CPU";
            }
            if (!strcasecmp(arg_string, "AI.MODELRUN")) {
                if (argc - 2 < argpos) {
                    RedisModule_WrongArity(ctx);
                    return REDISMODULE_ERR;
                }
                RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
                currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
                RAI_Model *mto;
                RedisModuleKey *modelKey;
                const int status = RAI_GetModelFromKeyspace(ctx, argv[argpos + 1], &modelKey, &mto,
                                                            REDISMODULE_READ);
                if (status == REDISMODULE_ERR) {
                    RAI_FreeRunInfo(rinfo);
                    return REDISMODULE_ERR;
                }
                currentOp->devicestr = mto->devicestr;
                currentOp->runkey = argv[argpos + 1];
                currentOp->mctx = RAI_ModelRunCtxCreate(mto);
            }
            if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
                if (argc - 3 < argpos) {
                    RedisModule_WrongArity(ctx);
                    return REDISMODULE_ERR;
                }
                RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
                currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
                RAI_Script *sto;
                RedisModuleKey *scriptKey;
                const int status = RAI_GetScriptFromKeyspace(ctx, argv[argpos + 1], &scriptKey,
                                                             &sto, REDISMODULE_READ);
                if (status == REDISMODULE_ERR) {
                    RAI_FreeRunInfo(rinfo);
                    return REDISMODULE_ERR;
                }
                currentOp->devicestr = sto->devicestr;
                const char *functionName = RedisModule_StringPtrLen(argv[argpos + 2], NULL);
                currentOp->runkey = argv[argpos + 1];
                currentOp->sctx = RAI_ScriptRunCtxCreate(sto, functionName);
            }
            if (RMAPI_FUNC_SUPPORTED(RedisModule_HoldString)) {
                RedisModule_HoldString(NULL, argv[argpos]);
            } else {
                RedisModule_RetainString(NULL, argv[argpos]);
            }
            RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
            currentOp->argv = array_append(currentOp->argv, argv[argpos]);
            currentOp->argc++;
        }
    }

    rinfo->dagOpCount = array_len(rinfo->dagOps);

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];
        if (currentOp == NULL)
            continue;
        int parse_result;
        switch (currentOp->commandType) {
        case REDISAI_DAG_CMD_TENSORSET:
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            break;
        case REDISAI_DAG_CMD_TENSORGET:
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            break;
        case REDISAI_DAG_CMD_MODELRUN:
            parse_result = RedisAI_Parse_ModelRun_RedisCommand(
                NULL, currentOp->argv, currentOp->argc, &(currentOp->mctx), &(currentOp->inkeys),
                &(currentOp->outkeys), &(currentOp->mctx->model), currentOp->err);
            if (parse_result < 0) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                return REDISMODULE_ERR;
            }
            break;
        case REDISAI_DAG_CMD_SCRIPTRUN:
            parse_result = RedisAI_Parse_ScriptRun_RedisCommand(
                NULL, currentOp->argv, currentOp->argc, &(currentOp->inkeys), &(currentOp->outkeys),
                &(currentOp->sctx->variadic), currentOp->err);
            if (parse_result < 0) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                return REDISMODULE_ERR;
            }
            break;
        }
    }

    if (rinfo->single_op_dag) {
        RAI_DagOp *op = rinfo->dagOps[0];
        RAI_Tensor *t;
        RedisModuleKey *key;
        for (size_t i = 0; i < array_len(op->inkeys); i++) {
            const char *inkey = RedisModule_StringPtrLen(op->inkeys[i], NULL);
            const int status =
                RAI_GetTensorFromKeyspace(ctx, op->inkeys[i], &key, &t, REDISMODULE_READ);
            if (status == REDISMODULE_ERR) {
                RedisModule_Log(ctx, "warning",
                                "on DAGRUN's LOAD could not load tensor %s from keyspace",
                                RedisModule_StringPtrLen(op->inkeys[i], NULL));
                return REDISMODULE_ERR;
            }
            RedisModule_CloseKey(key);
            char *dictKey = (char *)RedisModule_Alloc((strlen(inkey) + 5) * sizeof(char));
            sprintf(dictKey, "%s%04d", inkey, 1);
            AI_dictAdd(rinfo->dagTensorsContext, (void *)dictKey,
                       (void *)RAI_TensorGetShallowCopy(t));
            AI_dictAdd(rinfo->dagTensorsLoadedContext, (void *)dictKey, (void *)1);
            RedisModule_Free(dictKey);
        }

        for (size_t i = 0; i < array_len(op->outkeys); i++) {
            const char *outkey = RedisModule_StringPtrLen(op->outkeys[i], NULL);
            AI_dictAdd(rinfo->dagTensorsPersistedContext, (void *)outkey, (void *)1);
        }
    }

    // At this point, we have built a sequence of DAG operations, each with its own
    // input and output keys. The names of the keys will be used to look whether the
    // inputs to a DAG operation have all been realized by previous operations (or if
    // they are available as part of LOADed keys from keyspace).
    // This strategy is fine if keys are not aliased, that is, if a command's output
    // overwrites the key of a previous command. This would trick DAG operations into
    // thinking that their input is ready when it's not.
    // To overcome this, we make key names unique, so that names are not aliased. We
    // mangle the names by appending a numerical suffix ":0001". After computing, we
    // demangle the keys in order to persist them.

    AI_dict *mangled_tensors = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
    if (!mangled_tensors) {
        return REDISMODULE_ERR;
    }

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsLoadedContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            char *key = (char *)AI_dictGetKey(entry);
            char *demangled_key = RedisModule_Strdup(key);
            demangled_key[strlen(key) - 4] = 0;
            int *instance = RedisModule_Alloc(sizeof(int));
            *instance = 1;
            AI_dictAdd(mangled_tensors, (void *)demangled_key, (void *)instance);
            RedisModule_Free(demangled_key);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        RedisModuleString **mangled_inkeys =
            array_new(RedisModuleString *, array_len(currentOp->inkeys));
        for (long long j = 0; j < array_len(currentOp->inkeys); j++) {
            const char *key = RedisModule_StringPtrLen(currentOp->inkeys[j], NULL);
            AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
            if (!entry) {
                AI_dictRelease(mangled_tensors);
                RedisModule_ReplyWithError(ctx, "ERR INPUT key cannot be found in DAG");
                return REDISMODULE_ERR;
            }
            int *instance = AI_dictGetVal(entry);
            RedisModuleString *mangled_key =
                RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
            mangled_inkeys = array_append(mangled_inkeys, mangled_key);
        }

        RedisModuleString **mangled_outkeys =
            array_new(RedisModuleString *, array_len(currentOp->outkeys));
        for (long long j = 0; j < array_len(currentOp->outkeys); j++) {
            const char *key = RedisModule_StringPtrLen(currentOp->outkeys[j], NULL);
            AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
            int *instance = NULL;
            if (entry) {
                instance = AI_dictGetVal(entry);
                *instance += 1;
            } else {
                instance = RedisModule_Alloc(sizeof(int));
                *instance = 1;
                AI_dictAdd(mangled_tensors, (void *)key, (void *)instance);
            }
            RedisModuleString *mangled_key =
                RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
            mangled_outkeys = array_append(mangled_outkeys, mangled_key);
        }

        array_free(currentOp->inkeys);
        array_free(currentOp->outkeys);

        currentOp->inkeys = mangled_inkeys;
        currentOp->outkeys = mangled_outkeys;
    }

    AI_dict *mangled_persisted = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            char *key = (char *)AI_dictGetKey(entry);
            AI_dictEntry *mangled_entry = AI_dictFind(mangled_tensors, key);
            if (!mangled_entry) {
                AI_dictRelease(mangled_tensors);
                AI_dictRelease(mangled_persisted);
                RedisModule_ReplyWithError(ctx, "ERR PERSIST key cannot be found in DAG");
                AI_dictReleaseIterator(iter);
                RedisModule_ReplyWithError(ctx, "ERR PERSIST key cannot be found in DAG");
                return REDISMODULE_ERR;
            }
            int *instance = AI_dictGetVal(mangled_entry);
            RedisModuleString *mangled_key =
                RedisModule_CreateStringPrintf(ctx, "%s%04d", key, *instance);
            const char *mangled_key_str = RedisModule_StringPtrLen(mangled_key, NULL);
            AI_dictAdd(mangled_persisted, (void *)mangled_key_str, (void *)1);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    AI_dictRelease(rinfo->dagTensorsPersistedContext);
    rinfo->dagTensorsPersistedContext = mangled_persisted;

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(mangled_tensors);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            int *val = (int *)AI_dictGetVal(entry);
            RedisModule_Free(val);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }
    AI_dictRelease(mangled_tensors);
    mangled_tensors = NULL;

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        if (rinfo->dagOps[i]->devicestr == NULL) {
            rinfo->dagOps[i]->devicestr = "CPU";
        }
    }
    return REDISMODULE_OK;
}

// Add Shallow copies of the DAG run info to the devices' queues.
// Return REDISMODULE_OK in case of success, REDISMODULE_ERR if (at least) one insert op had failed.
static int DAG_InsertDAGToQueue(RedisAI_RunInfo *rinfo) {
    const char **devices = array_new(const char *, 10);

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        const char *devicestr = rinfo->dagOps[i]->devicestr;
        bool found = false;
        for (long long j = 0; j < array_len(devices); j++) {
            if (strcasecmp(devicestr, devices[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            devices = array_append(devices, devicestr);
        }
    }

    size_t ndevices = array_len(devices);
    RedisAI_RunInfo **rinfo_copies = array_new(RedisAI_RunInfo *, ndevices);

    for (long long i = 0; i < ndevices; i++) {
        RedisAI_RunInfo *rinfo_copy;
        RAI_ShallowCopyDagRunInfo(&rinfo_copy, rinfo);
        rinfo_copies = array_append(rinfo_copies, rinfo_copy);
    }

    for (long long i = 0; i < ndevices; i++) {
        RedisAI_RunInfo *rinfo_copy = rinfo_copies[i];
        for (long long j = 0; j < rinfo_copy->dagOpCount; j++) {
            if (strcasecmp(rinfo_copy->dagOps[j]->devicestr, devices[i]) == 0) {
                rinfo_copy->dagDeviceOps =
                    array_append(rinfo_copy->dagDeviceOps, rinfo_copy->dagOps[j]);
            }
        }
        rinfo_copy->dagDeviceOpCount = array_len(rinfo_copy->dagDeviceOps);
    }

    RunQueueInfo **run_queues_info = array_new(RunQueueInfo *, ndevices);
    for (long long i = 0; i < ndevices; i++) {
        const char *devicestr = devices[i];
        RunQueueInfo *run_queue_info = NULL;
        if (ensureRunQueue(devicestr, &run_queue_info) == REDISMODULE_ERR) {
            // A device run queue was not created properly, so we free everything,
            // set an error and finish.
            array_free(devices);
            for (int j = 0; j < ndevices; j++) {
                RAI_DagRunInfoFreeShallowCopy(rinfo_copies[j]);
            }
            array_free(rinfo_copies);
            array_free(run_queues_info);
            RAI_SetError(rinfo->err, RAI_EDAGRUN, "ERR Queue not initialized for device");
            rinfo->OnFinish((RedisAI_OnFinishCtx *)rinfo, rinfo->private_data);
            return REDISMODULE_ERR;
        }
        run_queues_info = array_append(run_queues_info, run_queue_info);
    }
    for (long long i = 0; i < ndevices; i++) {
        RedisAI_RunInfo *rinfo_copy = rinfo_copies[i];
        RunQueueInfo *run_queue_info = run_queues_info[i];
        gettimeofday(&rinfo_copy->queuingTime, NULL);

        pthread_mutex_lock(&run_queue_info->run_queue_mutex);
        queuePush(run_queue_info->run_queue, rinfo_copy);
        pthread_cond_signal(&run_queue_info->queue_condition_var);
        pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
    }

    array_free(devices);
    array_free(rinfo_copies);
    array_free(run_queues_info);
    return REDISMODULE_OK;
}

void DAG_ReplyAndUnblock(RedisAI_OnFinishCtx *ctx, void *private_data) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)ctx;
    if (rinfo->client)
        RedisModule_UnblockClient(rinfo->client, rinfo);
}

int RedisAI_ProcessDagRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                 int dagMode) {

    int flags = RedisModule_GetContextFlags(ctx);
    bool blocking_not_allowed = (flags & (REDISMODULE_CTX_FLAGS_MULTI | REDISMODULE_CTX_FLAGS_LUA));
    if (blocking_not_allowed)
        return RedisModule_ReplyWithError(
            ctx, "ERR Cannot run RedisAI command within a transaction or a LUA script");
    RedisAI_RunInfo *rinfo = NULL;
    if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(
            ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
        return REDISMODULE_ERR;
    }
    // Parse DAG string command and store the data in rinfo obj.
    int status = DAG_CommandParser(ctx, argv, argc, dagMode, &rinfo);
    if (status == REDISMODULE_ERR)
        return REDISMODULE_OK;
    // Block the client before adding rinfo to the run queues (sync call).
    rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, RedisAI_FreeData, 0);
    RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);
    rinfo->OnFinish = DAG_ReplyAndUnblock;
    return DAG_InsertDAGToQueue(rinfo);
}
