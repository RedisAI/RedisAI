#include "dag_execute.h"
#include "run_info.h"
#include "background_workers.h"
#include "util/string_utils.h"

int MangleTensorsNames(RedisAI_RunInfo *rinfo) {

    int res = REDISMODULE_ERR;
    AI_dict *mangled_tensors = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            RedisModuleString *key = (RedisModuleString *)AI_dictGetKey(entry);
            size_t key_len;
            const char *key_str = RedisModule_StringPtrLen(key, &key_len);
            RedisModuleString *demangled_key = RedisModule_CreateString(NULL, key_str, key_len - 4);
            int *instance = RedisModule_Alloc(sizeof(int));
            *instance = 1;
            AI_dictAdd(mangled_tensors, (void *)demangled_key, (void *)instance);
            RedisModule_FreeString(NULL, demangled_key);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        RedisModuleString **mangled_inkeys =
            array_new(RedisModuleString *, array_len(currentOp->inkeys));
        for (long long j = 0; j < array_len(currentOp->inkeys); j++) {
            RedisModuleString *key = currentOp->inkeys[j];
            AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
            if (!entry) {
                array_free(mangled_inkeys);
                RAI_SetError(rinfo->err, RAI_EDAGRUN, "ERR INPUT key cannot be found in DAG");
                goto cleanup;
            }
            int *instance = AI_dictGetVal(entry);
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));
            mangled_inkeys = array_append(mangled_inkeys, mangled_key);
        }

        RedisModuleString **mangled_outkeys =
            array_new(RedisModuleString *, array_len(currentOp->outkeys));
        for (long long j = 0; j < array_len(currentOp->outkeys); j++) {
            RedisModuleString *key = currentOp->outkeys[j];
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
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));
            mangled_outkeys = array_append(mangled_outkeys, mangled_key);
        }

        if (currentOp->inkeys) {
            for (size_t j = 0; j < array_len(currentOp->inkeys); j++) {
                RedisModule_FreeString(NULL, currentOp->inkeys[j]);
            }
            array_free(currentOp->inkeys);
        }

        if (currentOp->outkeys) {
            for (size_t j = 0; j < array_len(currentOp->outkeys); j++) {
                RedisModule_FreeString(NULL, currentOp->outkeys[j]);
            }
            array_free(currentOp->outkeys);
        }

        currentOp->inkeys = mangled_inkeys;
        currentOp->outkeys = mangled_outkeys;
    }

    AI_dict *mangled_persisted = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            RedisModuleString *key = (RedisModuleString *)AI_dictGetKey(entry);
            AI_dictEntry *mangled_entry = AI_dictFind(mangled_tensors, key);
            if (!mangled_entry) {
                AI_dictRelease(mangled_persisted);
                AI_dictReleaseIterator(iter);
                RAI_SetError(rinfo->err, RAI_EDAGRUN, "ERR PERSIST key cannot be found in DAG");
                goto cleanup;
            }
            int *instance = AI_dictGetVal(mangled_entry);
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));
            AI_dictAdd(mangled_persisted, (void *)mangled_key, (void *)1);
            RedisModule_FreeString(NULL, mangled_key);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    AI_dictRelease(rinfo->dagTensorsPersistedContext);
    rinfo->dagTensorsPersistedContext = mangled_persisted;

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        if (rinfo->dagOps[i]->devicestr == NULL) {
            rinfo->dagOps[i]->devicestr = "CPU";
        }
    }
    res = REDISMODULE_OK;

cleanup : {
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
    return res;
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
    if (MangleTensorsNames(rinfo) != REDISMODULE_OK) {
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

RAI_Tensor *RAI_DAGOutputTensor(RAI_OnFinishCtx *finish_ctx, size_t index) {
    size_t tensor_get_op_ind = -1;
    RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)finish_ctx;
    for (size_t i = 0; i < rinfo->dagOpCount; i++) {
        RAI_DagOp *op = rinfo->dagOps[i];
        if (op->commandType == REDISAI_DAG_CMD_TENSORGET) {
            tensor_get_op_ind++;
            if (tensor_get_op_ind == index) {
                RAI_Tensor *t;
                int res = RAI_getTensorFromLocalContext(rinfo->dagTensorsContext, op->inkeys[0], &t,
                                                        op->err);
                RedisModule_Assert(res == REDISMODULE_OK);
                return t;
            }
        }
    }
    return NULL;
}
