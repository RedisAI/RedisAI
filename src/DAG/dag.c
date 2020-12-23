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
#include "modelRun_ctx.h"
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
#include "dag_parser.h"
#include "util/string_utils.h"

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
        RedisModuleString *key_string = currentOp->outkeys[0];
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
    RedisModuleString *key_string = currentOp->inkeys[0];
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

static void Dag_LoadInputsToModelRunCtx(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {
    uint n_inkeys = array_len(currentOp->inkeys);
    uint n_outkeys = array_len(currentOp->outkeys);

    RAI_ContextReadLock(rinfo);

    RAI_Tensor *inputTensors[n_inkeys];
    for (uint i = 0; i < n_inkeys; i++) {
        RAI_Tensor *inputTensor;
        const int get_result = RAI_getTensorFromLocalContext(
            NULL, rinfo->dagTensorsContext, currentOp->inkeys[i], &inputTensor, currentOp->err);
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

    // Input and output names should match to the one specified when the model was set, only in TF.
    // For other backends, model->inputs and model->outputs is null.
    for (uint i = 0; i < n_inkeys; i++) {
        const char *opname = NULL;
        if (currentOp->mctx->model->inputs) {
            opname = currentOp->mctx->model->inputs[i];
        }
        RAI_ModelRunCtxAddInput(currentOp->mctx, opname, inputTensors[i]);
    }

    for (uint i = 0; i < n_outkeys; i++) {
        const char *opname = NULL;
        if (currentOp->mctx->model->outputs) {
            opname = currentOp->mctx->model->outputs[i];
        }
        RAI_ModelRunCtxAddOutput(currentOp->mctx, opname);
    }
}

static void Dag_StoreOutputsFromModelRunCtx(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp) {

    RAI_ContextReadLock(rinfo);
    const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RAI_Tensor *tensor = RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        AI_dictReplace(rinfo->dagTensorsContext, (void *)currentOp->outkeys[outputNumber], tensor);
    }
    RAI_ContextUnlock(rinfo);
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
        Dag_LoadInputsToModelRunCtx(rinfo, currentOp);

    RAI_ModelRunCtx *mctxs[1];
    mctxs[0] = currentOp->mctx;
    const long long start = ustime();
    int result = RAI_ModelRun(mctxs, 1, currentOp->err);
    const long long end = ustime();

    currentOp->duration_us = end - start;
    currentOp->result = result;
    if (result == REDISMODULE_ERR)
        return;
    if (rinfo->single_op_dag == 0)
        Dag_StoreOutputsFromModelRunCtx(rinfo, currentOp);
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

        // Get the needed tensors from the DAG local context. If the DAG originated
        // from a model run command, we are ready to execute.
        if (rinfo->single_op_dag == 0)
            Dag_LoadInputsToModelRunCtx(rinfo, currentOp);
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

        currentOp->duration_us = duration;
        currentOp->result = result;
        if (rinfo->single_op_dag == 0)
            Dag_StoreOutputsFromModelRunCtx(rinfo, currentOp);
    }
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
            NULL, rinfo->dagTensorsContext, currentOp->inkeys[i], &inputTensor, currentOp->err);
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
        RedisModuleString *key_string = currentOp->outkeys[outputNumber];
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        AI_dictReplace(rinfo->dagTensorsContext, (void *)key_string, tensor);
    }

    currentOp->result = result;
    currentOp->duration_us = end - start;

    RAI_ContextUnlock(rinfo);

    return;
}

size_t RAI_DagOpBatchSize(RAI_DagOp *op, RedisAI_RunInfo *rinfo) {
    if (op->mctx == NULL) {
        return -1;
    }

    size_t ninputs = array_len(op->inkeys);
    int batchsize = 0;

    if (!rinfo->single_device_dag) {
        RAI_ContextReadLock(rinfo);
    }
    for (size_t i = 0; i < ninputs; i++) {
        RAI_Tensor *input;
        if (rinfo->single_op_dag) {
            input = op->mctx->inputs[i].tensor;
        } else {
            RAI_getTensorFromLocalContext(NULL, rinfo->dagTensorsContext, op->inkeys[i], &input,
                                          op->err);
        }
        // We are expecting input != NULL, because we only reach this function if all inputs
        // are available in context for the current dagOp. We could be more defensive
        // eventually.
        assert(input != NULL);

        if (i == 0) {
            batchsize = RAI_TensorDim(input, 0);
            continue;
        }
        if (batchsize != RAI_TensorDim(input, 0)) {
            batchsize = 0;
            break;
        }
    }
    if (!rinfo->single_device_dag) {
        RAI_ContextUnlock(rinfo);
    }
    return batchsize;
}

int RAI_DagOpBatchable(RAI_DagOp *op1, RedisAI_RunInfo *rinfo1, RAI_DagOp *op2,
                       RedisAI_RunInfo *rinfo2) {

    if (op1->mctx == NULL || op2->mctx == NULL) {
        return 0;
    }
    if (op1->mctx->model != op2->mctx->model) {
        return 0;
    }
    const int ninputs1 = array_len(op1->inkeys);
    const int ninputs2 = array_len(op2->inkeys);

    if (ninputs1 != ninputs2) {
        return 0;
    }
    if (!rinfo1->single_device_dag) {
        RAI_ContextReadLock(rinfo1);
    }
    if (!rinfo2->single_device_dag) {
        RAI_ContextReadLock(rinfo2);
    }
    for (int i = 0; i < ninputs1; i++) {
        RAI_Tensor *input1;
        if (rinfo1->single_op_dag == 1) {
            input1 = op1->mctx->inputs[i].tensor;
        } else {
            RAI_getTensorFromLocalContext(NULL, rinfo1->dagTensorsContext, op1->inkeys[i], &input1,
                                          op1->err);
        }
        RAI_Tensor *input2;
        if (rinfo2->single_op_dag == 1) {
            input2 = op2->mctx->inputs[i].tensor;
        } else {
            RAI_getTensorFromLocalContext(NULL, rinfo2->dagTensorsContext, op2->inkeys[i], &input2,
                                          op2->err);
        }
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
            long long dim1 = RAI_TensorDim(input1, j);
            long long dim2 = RAI_TensorDim(input2, j);
            if (dim1 != dim2) {
                return 0;
            }
        }
    }
    if (!rinfo1->single_device_dag) {
        RAI_ContextUnlock(rinfo1);
    }
    if (!rinfo2->single_device_dag) {
        RAI_ContextUnlock(rinfo2);
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
    *currentOpReady = 1;
    // If this is a single op dag, the op is definitely ready.
    if (rinfo->single_op_dag == 1)
        return;

    uint n_inkeys = array_len(currentOp_->inkeys);
    RAI_ContextReadLock(rinfo);

    for (int i = 0; i < n_inkeys; i++) {
        if (AI_dictFind(rinfo->dagTensorsContext, currentOp_->inkeys[i]) == NULL) {
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
    if (!op->mctx)
        return;

    *batchsize = op->mctx->model->opts.batchsize;
    *minbatchsize = op->mctx->model->opts.minbatchsize;
    *minbatchtimeout = op->mctx->model->opts.minbatchtimeout;
    *inbatchsize = RAI_DagOpBatchSize(op, rinfo);
}

void RedisAI_DagOpBatchingMatch(RedisAI_RunInfo *rinfo1, RAI_DagOp *op1, RedisAI_RunInfo *rinfo2,
                                RAI_DagOp *op2, int *batched, size_t *inbatchsize) {
    *batched = 0;
    *inbatchsize = 0;

    if (op2->mctx) {
        int match = RAI_DagOpBatchable(op1, rinfo1, op2, rinfo2);

        if (match) {
            *batched = 1;
            *inbatchsize = RAI_DagOpBatchSize(op2, rinfo2);
        }
    }
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

static int _StoreTensorInKeySpace(RedisModuleCtx *ctx, RAI_Tensor *tensor,
                                  RedisModuleString *persist_key_name, bool mangled_name) {

    int ret = REDISMODULE_ERR;
    RedisModuleKey *key;
    size_t persist_key_len;
    const char *persist_key_str = RedisModule_StringPtrLen(persist_key_name, &persist_key_len);

    RedisModuleString *demangled_key_name;
    if (mangled_name) {
        demangled_key_name = RedisModule_CreateString(NULL, persist_key_str, persist_key_len - 4);
    } else {
        demangled_key_name = RedisModule_CreateString(NULL, persist_key_str, persist_key_len);
    }

    const int status =
        RAI_OpenKey_Tensor(ctx, demangled_key_name, &key, REDISMODULE_READ | REDISMODULE_WRITE);
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
        goto clean_up;
    } else {
        if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType,
                                           RAI_TensorGetShallowCopy(tensor)) != REDISMODULE_OK) {
            RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
            goto clean_up;
        }
    }
    ret = REDISMODULE_OK;

clean_up:
    RedisModule_CloseKey(key);
    RedisAI_ReplicateTensorSet(ctx, demangled_key_name, tensor);
    RedisModule_FreeString(NULL, demangled_key_name);
    return ret;
}

static void PersistTensors(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo) {
    AI_dictIterator *persist_iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
    AI_dictEntry *persist_entry = AI_dictNext(persist_iter);
    while (persist_entry) {
        RedisModuleString *persist_key_name = AI_dictGetKey(persist_entry);

        AI_dictEntry *tensor_entry = AI_dictFind(rinfo->dagTensorsContext, persist_key_name);

        if (tensor_entry) {
            RAI_Tensor *tensor = AI_dictGetVal(tensor_entry);

            if (tensor == NULL) {
                persist_entry = AI_dictNext(persist_iter);
                continue;
            }
            if (_StoreTensorInKeySpace(ctx, tensor, persist_key_name, true) == REDISMODULE_ERR)
                rinfo->dagReplyLength++;

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
                RedisModuleString *localcontext_key_name = AI_dictGetKey(local_entry);
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
}

static void ModelSingleOp_PersistTensors(RedisModuleCtx *ctx, RAI_DagOp *op) {
    const size_t noutputs = RAI_ModelRunCtxNumOutputs(op->mctx);
    for (size_t outputNumber = 0; outputNumber < noutputs; outputNumber++) {
        RAI_Tensor *tensor = RAI_ModelRunCtxOutputTensor(op->mctx, outputNumber);
        tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL;
        if (tensor)
            _StoreTensorInKeySpace(ctx, tensor, op->outkeys[outputNumber], false);
    }
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
            RAI_GetRunStats(currentOp->runkey, &rstats);
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

    // TODO: Take care of script single op
    if (rinfo->single_op_dag == 0 || rinfo->dagOps[0]->commandType == REDISAI_DAG_CMD_SCRIPTRUN) {
        // Save the required tensors in redis key space.
        PersistTensors(ctx, rinfo);
        if (rinfo->single_op_dag == 0)
            RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
    } else {
        ModelSingleOp_PersistTensors(ctx, rinfo->dagOps[0]);
    }

    RAI_FreeRunInfo(rinfo);

    return REDISMODULE_OK;
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

void RunInfo_FreeData(RedisModuleCtx *ctx, void *rinfo) {}

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
    RedisModule_Log(ctx, "warning", "Blocked client %p disconnected!", (void *)bc);
}

// Add Shallow copies of the DAG run info to the devices' queues.
// Return REDISMODULE_OK in case of success, REDISMODULE_ERR if (at least) one insert op had
// failed.
int DAG_InsertDAGToQueue(RedisAI_RunInfo *rinfo) {
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
    if (ndevices == 1)
        rinfo->single_device_dag = 1;
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

void Dag_PopulateOp(RAI_DagOp *currentOp, void *rctx, RedisModuleString **inkeys,
                    RedisModuleString **outkeys, RedisModuleString *runkey) {

    if (currentOp->commandType == REDISAI_DAG_CMD_MODELRUN) {
        currentOp->mctx = (RAI_ModelRunCtx *)rctx;
        currentOp->devicestr = currentOp->mctx->model->devicestr;
    } else {
        assert(currentOp->commandType == REDISAI_DAG_CMD_SCRIPTRUN);
        currentOp->sctx = (RAI_ScriptRunCtx *)rctx;
        currentOp->devicestr = currentOp->sctx->script->devicestr;
    }

    currentOp->inkeys = inkeys;
    currentOp->outkeys = outkeys;
    currentOp->runkey = runkey;
}
