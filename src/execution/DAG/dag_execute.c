#include <sys/time.h>
#include <execution/run_queue_info.h>
#include "dag_execute.h"
#include "util/string_utils.h"
#include "execution/run_info.h"
#include "execution/background_workers.h"

int ValidatePersistKeys(RedisAI_RunInfo *rinfo, AI_dict *tensorsNamesToInd,
                        AI_dict *persistTensorsNames) {

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(persistTensorsNames);
        AI_dictEntry *persist_entry;
        while ((persist_entry = AI_dictNext(iter))) {
            RedisModuleString *persist_key = (RedisModuleString *)AI_dictGetKey(persist_entry);
            AI_dictEntry *entry = AI_dictFind(tensorsNamesToInd, persist_key);
            if (!entry) {
                RAI_SetError(rinfo->err, RAI_EDAGRUN, "ERR PERSIST key cannot be found in DAG");
                AI_dictReleaseIterator(iter);
                return REDISMODULE_ERR;
            }
            size_t index = (size_t)AI_dictGetVal(entry);
            AI_dictReplace(persistTensorsNames, (void *)persist_key, (void *)index);
        }
        AI_dictReleaseIterator(iter);
    }
    return REDISMODULE_OK;
}

int MapTensorsKeysToIndices(RedisAI_RunInfo *rinfo, AI_dict *tensorsNamesToInd) {

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        for (long long j = 0; j < array_len(currentOp->inkeys); j++) {
            RedisModuleString *key = currentOp->inkeys[j];
            AI_dictEntry *entry = AI_dictFind(tensorsNamesToInd, key);
            if (!entry) {
                RAI_SetError(rinfo->err, RAI_EDAGRUN, "ERR INPUT key cannot be found in DAG");
                return REDISMODULE_ERR;
            }
            size_t ind = (size_t)AI_dictGetVal(entry);
            currentOp->inkeys_indices = array_append(currentOp->inkeys_indices, ind);
        }

        for (long long j = 0; j < array_len(currentOp->outkeys); j++) {
            RedisModuleString *key = currentOp->outkeys[j];
            size_t ind = array_len(rinfo->dagSharedTensors);

            // Add a new empty place holder in the array for an output tensor.
            // If this is a TENSORSET op, the tensor is already realized.
            if (currentOp->commandType == REDISAI_DAG_CMD_TENSORSET) {
                RAI_Tensor *t = RAI_TensorGetShallowCopy(currentOp->outTensor);
                rinfo->dagSharedTensors = array_append(rinfo->dagSharedTensors, t);
            } else {
                rinfo->dagSharedTensors = array_append(rinfo->dagSharedTensors, NULL);
            }
            currentOp->outkeys_indices = array_append(currentOp->outkeys_indices, ind);
            AI_dictReplace(tensorsNamesToInd, (void *)key, (void *)ind);
        }
    }
    return REDISMODULE_OK;
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
        const char *device_str = devices[i];
        RunQueueInfo *run_queue_info = RunQueue_GetInfo(device_str);
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

int RAI_DAGRun(RAI_DAGRunCtx *run_info, RAI_OnFinishCB DAGAsyncFinish, void *private_data,
               RAI_Error *err) {

    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    if (rinfo->dagOpCount < 1) {
        RAI_SetError(err, RAI_EDAGRUN, "ERR DAG is empty");
        return REDISMODULE_ERR;
    }
    // Make the inkeys and outkeys of the DAG ops unique, to ensure that the operations
    // will be execute in the right order.
    if (MapTensorsKeysToIndices(rinfo, rinfo->tensorsNamesToIndices) != REDISMODULE_OK) {
        RAI_SetError(err, rinfo->err->code, rinfo->err->detail);
        return REDISMODULE_ERR;
    }
    rinfo->OnFinish = (RedisAI_OnFinishCB)DAGAsyncFinish;
    rinfo->private_data = private_data;
    if (DAG_InsertDAGToQueue(rinfo) != REDISMODULE_OK) {
        RAI_SetError(err, rinfo->err->code, rinfo->err->detail);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

size_t RAI_DAGNumOutputs(RAI_OnFinishCtx *finish_ctx) {
    size_t n_outputs = 0;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)finish_ctx;
    for (size_t i = 0; i < rinfo->dagOpCount; i++) {
        if (rinfo->dagOps[i]->commandType == REDISAI_DAG_CMD_TENSORGET) {
            n_outputs++;
        }
    }
    return n_outputs;
}

const RAI_Tensor *RAI_DAGOutputTensor(RAI_OnFinishCtx *finish_ctx, size_t index) {
    size_t tensor_get_op_ind = -1;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)finish_ctx;

    for (size_t i = 0; i < rinfo->dagOpCount; i++) {
        RAI_DagOp *op = rinfo->dagOps[i];
        if (op->commandType == REDISAI_DAG_CMD_TENSORGET) {
            tensor_get_op_ind++;
            if (tensor_get_op_ind == index) {
                return Dag_GetTensorFromGlobalCtx(rinfo, op->inkeys_indices[0]);
            }
        }
    }
    return NULL;
}

bool RAI_DAGRunError(RAI_OnFinishCtx *finish_ctx) {
    return *((RedisAI_RunInfo *)finish_ctx)->dagError;
}

const RAI_Error *RAI_DAGGetError(RAI_OnFinishCtx *finish_ctx) {
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)finish_ctx;
    return rinfo->err;
}
