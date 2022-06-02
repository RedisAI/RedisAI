#include "sys/time.h"
#include "run_info.h"
#include "run_queue_info.h"
#include "execution/DAG/dag.h"

/* Define for RedisAI thread name setter */
#ifdef __linux__
#define RAI_PTHREAD_SETNAME(name) pthread_setname_np(pthread_self(), name)
#else
#if (defined __NetBSD__ || defined __FreeBSD__ || defined __OpenBSD__)
#include <pthread_np.h>
#define RAI_PTHREAD_SETNAME(name) pthread_set_name_np(pthread_self(), name)
#else
#if (defined __APPLE__ && defined(MAC_OS_X_VERSION_10_7))
int pthread_setname_np(const char *name);
#include <pthread.h>
#define RAI_PTHREAD_SETNAME(name) pthread_setname_np(name)
#else
#define RAI_PTHREAD_SETNAME(name)
#endif
#endif
#endif

uintptr_t LastThreadId;      // Last number given as thread id for BG threads running currently.
pthread_key_t ThreadIdKey;   // Key to hold thread id in its local storage.
unsigned int BGWorkersCount; // Total number of BG threads spawned.

/**
 * @brief Save the id for some working thread in thread local storage.
 */
static void _BGWorker_SaveThreadId() {
    // Let the current thread have the next available id, and increase the counter.
    long id_value = __atomic_add_fetch(&LastThreadId, 1, __ATOMIC_RELAXED);
    // Convert the id value to a pointer and store it the thread local storage.
    // First id is 1, so we won't confuse with NULL (which is the error return value)
    pthread_setspecific(ThreadIdKey, (const void *)id_value);
}

/**
 * @brief In case a DAG Op can express a MINBATCHSIZE > 0 with a MINBATCHTIMEOUT
 * in milliseconds, we will use a timedwait of one millisecond to evaluate
 * whether the run needs to trigger in case nothing else happens on the queue.
 * Possible optimization: opt for a longer timeout if there's no
 * minbatchtimeout involved.
 */
static void _BGThread_Wait(RunQueueInfo *run_queue_info) {
    struct timeval now;
    gettimeofday(&now, NULL);

    struct timespec absTimeout;
    absTimeout.tv_sec = now.tv_sec;
    absTimeout.tv_nsec = (now.tv_usec + 1000) * 1000; // 1 millisecond

    pthread_cond_timedwait(&run_queue_info->queue_condition_var, &run_queue_info->run_queue_mutex,
                           &absTimeout);
}

static void _BGThread_SaveStats(RedisAI_RunInfo *rinfo) {
    for (size_t i = 0; i < rinfo->dagOpCount; i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        if (currentOp->commandType == REDISAI_DAG_CMD_MODELRUN ||
            currentOp->commandType == REDISAI_DAG_CMD_SCRIPTRUN) {
            if (currentOp->result == REDISMODULE_ERR) {
                RAI_StatsAddDataPoint(RAI_ExecutionCtx_GetStats(currentOp->ectx), 0, 1, 1, 0);
            } else if (currentOp->result == REDISMODULE_OK) {
                unsigned long batch_size = 1;
                if (currentOp->commandType == REDISAI_DAG_CMD_MODELRUN) {
                    RAI_Tensor *t = NULL;
                    if (RAI_ExecutionCtx_NumOutputs(currentOp->ectx) > 0) {
                        t = RAI_ExecutionCtx_GetOutput(currentOp->ectx, 0);
                    }
                    if (t) {
                        batch_size = RAI_TensorDim(t, 0);
                    } else {
                        batch_size = 0;
                    }
                }
                RAI_StatsAddDataPoint(RAI_ExecutionCtx_GetStats(currentOp->ectx),
                                      currentOp->duration_us, 1, 0, batch_size);
            }
        }
    }
}

static void _BGThread_RinfoFinish(RedisAI_RunInfo *rinfo) {
    RedisAI_RunInfo *orig = rinfo->orig_copy;
    uint dagRefCount = RAI_DagRunInfoFreeShallowCopy(rinfo);
    if (dagRefCount == 0) {
        // Save stats for every DAG execute operation.
        _BGThread_SaveStats(orig);
        RedisAI_OnFinishCtx *finish_ctx = orig;
        orig->OnFinish(finish_ctx, orig->private_data);
    }
}

static bool _BGThread_IsRInfoTimedOut(RedisAI_RunInfo *rinfo) {

    if (RedisAI_DagTimeout(rinfo)) {
        return true;
    }
    if (rinfo->timeout > 0) {
        struct timeval now, sub;
        gettimeofday(&now, NULL);
        timersub(&now, &rinfo->queuingTime, &sub);
        size_t time_msec = sub.tv_sec * 1000 + sub.tv_usec / 1000;
        if (time_msec > rinfo->timeout) {
            RedisAI_DagSetTimeout(rinfo);
            return true;
        }
    }
    return false;
}

static int *_BGThread_ExecutionFinish(RedisAI_RunInfo **batch_rinfo) {
    int *unfinished_rinfos_indices = array_new(int, array_len(batch_rinfo));
    for (int i = array_len(batch_rinfo) - 1; i >= 0; i--) {
        RedisAI_RunInfo *rinfo = batch_rinfo[i];
        rinfo->dagDeviceCompleteOpCount += 1;
        __atomic_add_fetch(rinfo->dagCompleteOpCount, 1, __ATOMIC_RELAXED);
        if (RedisAI_DagDeviceComplete(rinfo) || RedisAI_DagError(rinfo) ||
            RedisAI_DagTimeout(rinfo)) {
            _BGThread_RinfoFinish(rinfo);
        } else {
            unfinished_rinfos_indices = array_append(unfinished_rinfos_indices, i);
        }
    }
    return unfinished_rinfos_indices;
}

static void _BGThread_Execute(RunQueueInfo *run_queue_info, RedisAI_RunInfo **batch_rinfo) {
    uint n_rinfo = array_len(batch_rinfo);
    if (n_rinfo != 0) {
        bool batched_run = n_rinfo > 1;
        // For simplicity, we call into different functions whether the run
        // is batched or not
        if (batched_run) {
            RedisAI_BatchedDagRunSessionStep(batch_rinfo, run_queue_info->device_str);
        } else {
            RedisAI_DagRunSessionStep(batch_rinfo[0], run_queue_info->device_str);
        }
    }
}

static RedisAI_RunInfo **_BGThread_BatchOperations(RunQueueInfo *run_queue_info,
                                                   RedisAI_RunInfo *rinfo,
                                                   RedisAI_RunInfo **batch_rinfo,
                                                   bool *batchReady) {
    // Since the current op can be batched, then we collect info on batching, namely
    // - batchsize
    // - minbatchsize
    // - minbatchtimeout
    // - actual size of the input along the batch (0-th) dimension
    RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);
    size_t batchsize, minbatchsize, minbatchtimeout, inbatchsize;
    RedisAI_DagOpBatchInfo(rinfo, currentOp, &batchsize, &minbatchsize, &minbatchtimeout,
                           &inbatchsize);

    // Get the size of the batch so far, that is, the size of the first input
    // tensor in the 0-th dimension
    size_t current_batchsize = inbatchsize;

    // If the size is zero or if it already exceeds the desired batch size
    // then stop searching
    if (current_batchsize == 0 || current_batchsize >= batchsize) {
        return batch_rinfo;
    }

    // Set the batch to be ready by default (optimistic), change it during run.
    *batchReady = true;
    bool timeout = false;
    // If minbatchsize has been set and we are not past it, we check
    // if the timeout for min batch has expired, in which case we proceed
    // anyway
    if (minbatchsize > 0 && minbatchtimeout > 0) {
        struct timeval now, sub;
        gettimeofday(&now, NULL);

        timersub(&now, &rinfo->queuingTime, &sub);
        size_t time_msec = sub.tv_sec * 1000 + sub.tv_usec / 1000;

        if (time_msec > minbatchtimeout) {
            timeout = true;
        }
    }

    // Get the next item in the queue
    queueItem *next_item = queueFront(run_queue_info->run_queue);

    // While we don't reach the end of the queue
    while (next_item != NULL && !timeout) {
        // Get the next run info
        RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;

        // If the next item is batchable, that is, if it is a model, if it
        // is a call to the same model, and if the size of the inputs except
        // the 0-th dimension match, then go on, otherwise continue to the
        // next item in the queue
        RAI_DagOp *nextOp = RedisAI_DagCurrentOp(next_rinfo);

        bool nextOpReady, nextOpBatchable;
        RedisAI_DagCurrentOpInfo(next_rinfo, &nextOpReady, &nextOpBatchable);

        if (nextOpReady == 0 || nextOpBatchable == 0) {
            next_item = queueNext(next_item);
            continue;
        }

        int batched = 0;
        size_t next_batchsize = 0;
        RedisAI_DagOpBatchingMatch(rinfo, currentOp, next_rinfo, nextOp, &batched, &next_batchsize);

        if (batched == 0) {
            next_item = queueNext(next_item);
            continue;
        }

        // If all previous checks pass, then keep track of the item
        // in the list of evicted items
        queueItem *tmp = queueNext(next_item);
        queueItem *evicted = queueEvict(run_queue_info->run_queue, next_item);
        RedisModule_Free(evicted);
        next_item = tmp;
        batch_rinfo = array_append(batch_rinfo, next_rinfo);

        // Update the batchsize and go to the next item to see if
        // there's anything else to batch
        current_batchsize += next_batchsize;

        // If the new batch size would exceed the prescribed batch
        // size, then quit searching.
        // Here we could consider searching further down the queue.
        if (current_batchsize >= batchsize) {
            break;
        }

        // If minbatchsize has been set and we are not past it, we check
        // if the timeout for min batch has expired, in which case we proceed
        // anyway
        if (minbatchsize > 0 && minbatchtimeout > 0) {
            struct timeval now, sub;
            gettimeofday(&now, NULL);

            timersub(&now, &rinfo->queuingTime, &sub);
            size_t time_msec = sub.tv_sec * 1000 + sub.tv_usec / 1000;

            if (time_msec > minbatchtimeout) {
                timeout = true;
            }
        }
    }
    if (minbatchsize != 0 && current_batchsize < minbatchsize) {
        // The batch is ready with respect to minbatch only if there was a timeout.
        *batchReady = timeout;
    }
    return batch_rinfo;
}

static bool _BGThread_PrepareExecution(RunQueueInfo *run_queue_info, RedisAI_RunInfo *rinfo,
                                       RedisAI_RunInfo ***batch_rinfo) {
    // Get if the operation is ready and bacthable
    bool currentOpReady, currentOpBatchable;
    RedisAI_DagCurrentOpInfo(rinfo, &currentOpReady, &currentOpBatchable);
    if (currentOpReady) {
        *batch_rinfo = array_append(*batch_rinfo, rinfo);
    } else {
        // Op is not ready - push back to queue and continue the loop.
        queuePush(run_queue_info->run_queue, rinfo);
        return false;
    }

    if (currentOpBatchable) {
        bool batchReady = true;
        *batch_rinfo = _BGThread_BatchOperations(run_queue_info, rinfo, *batch_rinfo, &batchReady);
        if (!batchReady) {
            // Batch is not ready - batch size didn't match the expectations from
            // minbatchsize
            for (int i = array_len(*batch_rinfo) - 1; i >= 0; i--) {
                queuePush(run_queue_info->run_queue, (*batch_rinfo)[i]);
            }
            return false;
        }
    }
    return true;
}

long BGWorker_GetThreadId() {
    void *thread_id = pthread_getspecific(ThreadIdKey);

    // Return the 0 based id, if thread_id was NULL, we return -1 to indicates that
    // the caller is not RedisAI thread.
    return (long)(thread_id)-1;
}

uintptr_t BGWorker_GetThreadsCount() { return BGWorkersCount; }

void *BGWorker_ThreadMain(void *arg) {
    _BGWorker_SaveThreadId();
    RunQueueInfo *run_queue_info = (RunQueueInfo *)arg;
    RedisAI_RunInfo **batch_rinfo = array_new(RedisAI_RunInfo *, 1);
    pthread_mutex_lock(&run_queue_info->run_queue_mutex);

    while (true) {
        _BGThread_Wait(run_queue_info);
        // This is the length of the queue for this particular device
        // (see run_queue_info->devicestr).
        // There might be more than one thread operating on the same
        // queue, according to the THREADS_PER_QUEUE config variable.
        while (queueFront(run_queue_info->run_queue)) {
            array_clear(batch_rinfo);
            // We first peek the front of the queue
            queueItem *item = queuePop(run_queue_info->run_queue);
            RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)item->value;
            RedisModule_Free(item);
            // In case of timeout or error - skip execution.
            bool skip_execution = _BGThread_IsRInfoTimedOut(rinfo) || RedisAI_DagError(rinfo);
            // Prepare to execution, if the op or the batch is not ready, exit
            // the loop, give a chance to new tasks to submit.
            if (!skip_execution &&
                !_BGThread_PrepareExecution(run_queue_info, rinfo, &batch_rinfo)) {
                break;
            }
            // Run the computation step (batched or not)
            // We're done with the queue here, items have been evicted so we can
            // safely unlock the queue mutex, to allow other threads to operate
            // on the same queue. The evicted items at this point are only visible
            // to this worker.
            pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
            if (!skip_execution) {
                _BGThread_Execute(run_queue_info, batch_rinfo);
            } else {
                // If we are skipping the execution due to dag error or timeout,
                // we consider the batch as contains this single dag when finish.
                batch_rinfo = array_append(batch_rinfo, rinfo);
            }
            // For every DAG in the batch: if the entire DAG run is complete,
            // call the on finish callback. Otherwise, save the DAG index in
            // the batch_rinfo array, so we reinsert the DAG to the queue
            // (after acquiring the queue lock).
            int *unfinished_rinfo_indices = _BGThread_ExecutionFinish(batch_rinfo);
            pthread_mutex_lock(&run_queue_info->run_queue_mutex);

            // Reinsert the unfinished DAG's run info to the queue.
            for (size_t i = 0; i < array_len(unfinished_rinfo_indices); i++) {
                queuePushFront(run_queue_info->run_queue, batch_rinfo[unfinished_rinfo_indices[i]]);
            }
            array_free(unfinished_rinfo_indices);
        }
    }
    array_free(batch_rinfo);
}
