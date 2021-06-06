/**
 * background_workers.c
 *
 * Contains the structure to manage the per-device queues, used for decoupling
 * the work from the main thread to the background worker threads. For each of
 * the incoming ModelRun, ScriptRun, and DagRun commands, the request is queued
 * and evaded asynchronously to one the device queues.
 *
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include "string_utils.h"
#include "backends/backends.h"
#include <bits/unistd_ext.h>
#include "redisai.h"
#include "run_info.h"
#include "background_workers.h"
#include "onnx_timeout.h"

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

void *RedisAI_Run_ThreadMain(void *arg);

RunQueueInfo *CreateRunQueue(const char *device_str) {

    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    String_ToUpper(device_str, upper_device_str, &device_str_len);

    // Create new run queue and initialize its inner fields.
    RunQueueInfo *run_queue_info = RedisModule_Alloc(sizeof(RunQueueInfo));
    run_queue_info->run_queue = queueCreate();
    run_queue_info->device_str = RedisModule_Strdup(upper_device_str);
    pthread_cond_init(&(run_queue_info->queue_condition_var), NULL);
    pthread_mutex_init(&(run_queue_info->run_queue_mutex), NULL);
    run_queue_info->threads =
        (pthread_t *)RedisModule_Alloc(sizeof(pthread_t) * ThreadPoolSizePerQueue);

    // Save device with its associate run queue info in the dictionary.
    if (AI_dictAdd(RunQueues, upper_device_str, run_queue_info) != DICT_OK) {
        RunQueueInfoFree(run_queue_info);
        return NULL;
    }

    // Create worker threads.
    for (int i = 0; i < ThreadPoolSizePerQueue; i++) {
        if (pthread_create(&(run_queue_info->threads[i]), NULL, RedisAI_Run_ThreadMain,
                           run_queue_info) != 0) {
            AI_dictDelete(RunQueues, upper_device_str);
            RunQueueInfoFree(run_queue_info);
            return NULL;
        }
    }

    // Add the new device worker threads to onnx run sessions monitoring.
    if (RAI_backends.onnx.add_new_device) {
        RAI_backends.onnx.add_new_device(device_str);
    }
    return run_queue_info;
}

RunQueueInfo *GetRunQueueInfo(const char *device_str) {
    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    String_ToUpper(device_str, upper_device_str, &device_str_len);
    AI_dictEntry *entry = AI_dictFind(RunQueues, upper_device_str);
    RedisModule_Assert(entry != NULL);
    return AI_dictGetVal(entry);
}

bool IsRunQueueExists(const char *device_str) {
    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    String_ToUpper(device_str, upper_device_str, &device_str_len);
    if (AI_dictFind(RunQueues, upper_device_str) == NULL) {
        return false;
    }
    return true;
}

uintptr_t GetThreadId() { return *(uintptr_t *)pthread_getspecific(ThreadIdKey); }

void RunQueueInfoFree(RunQueueInfo *run_queue_info) {
    RedisModule_Assert(queueLength(run_queue_info->run_queue) == 0);
    RedisModule_Free(run_queue_info->run_queue);
    RedisModule_Free(run_queue_info->device_str);

    // Wait for workers to exit and free the pool.
    for (int i = 0; i < ThreadPoolSizePerQueue; i++) {
        RedisModule_Assert(pthread_join(run_queue_info->threads[i], NULL) == 0);
        RedisModule_Free(run_queue_info->threads);
    }
    pthread_mutex_destroy(&(run_queue_info->run_queue_mutex));
    pthread_cond_destroy(&(run_queue_info->queue_condition_var));
    RedisModule_Free(run_queue_info);
}

/**
 * @brief Save the id for some working thread in thread local storage.
 */
static void _SaveThreadId() {
    uintptr_t *id_value = RedisModule_Alloc(sizeof(uintptr_t));
    // Let the current thread have the next available id, and increase the counter.
    *id_value = __atomic_fetch_add(&BGWorkersCounter, 1, __ATOMIC_RELAXED);
    pthread_setspecific(ThreadIdKey, id_value);
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

static void _BGThread_RinfoFinish(RedisAI_RunInfo *rinfo) {
    RedisAI_RunInfo *orig = rinfo->orig_copy;
    uint dagRefCount = RAI_DagRunInfoFreeShallowCopy(rinfo);
    if (dagRefCount == 0) {
        RedisAI_OnFinishCtx *finish_ctx = orig;
        orig->OnFinish(finish_ctx, orig->private_data);
    }
}

static bool _BGThread_IsRInfoTimedOut(RedisAI_RunInfo *rinfo) {
    bool timedOut = false;
    if (rinfo->timeout > 0) {
        timedOut = __atomic_load_n(rinfo->timedOut, __ATOMIC_RELAXED);

        if (!timedOut) {
            struct timeval now, sub;
            gettimeofday(&now, NULL);
            timersub(&now, &rinfo->queuingTime, &sub);
            size_t time_msec = sub.tv_sec * 1000 + sub.tv_usec / 1000;

            if (time_msec > rinfo->timeout) {
                timedOut = true;
                __atomic_store_n(rinfo->timedOut, timedOut, __ATOMIC_RELAXED);
            }
        }
    }
    return timedOut;
}

static void _BGThread_ExecutionFinish(RunQueueInfo *run_queue_info, RedisAI_RunInfo **batch_rinfo) {
    for (int i = array_len(batch_rinfo) - 1; i >= 0; i--) {
        RedisAI_RunInfo *rinfo = batch_rinfo[i];
        rinfo->dagDeviceCompleteOpCount += 1;
        __atomic_add_fetch(rinfo->dagCompleteOpCount, 1, __ATOMIC_RELAXED);
        if (RedisAI_DagDeviceComplete(rinfo) || RedisAI_DagError(rinfo)) {
            _BGThread_RinfoFinish(rinfo);
        } else {
            queuePushFront(run_queue_info->run_queue, rinfo);
        }
    }
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

void *RedisAI_Run_ThreadMain(void *arg) {
    _SaveThreadId();
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
            // In case of timeout.
            if (_BGThread_IsRInfoTimedOut(rinfo)) {
                _BGThread_RinfoFinish(rinfo);
                continue;
            }

            if (RedisAI_DagError(rinfo)) {
                _BGThread_RinfoFinish(rinfo);
                continue;
            }

            // Get if the operation is ready and bacthable
            bool currentOpReady, currentOpBatchable;
            RedisAI_DagCurrentOpInfo(rinfo, &currentOpReady, &currentOpBatchable);
            if (currentOpReady) {
                batch_rinfo = array_append(batch_rinfo, rinfo);
            } else {
                // Op is not ready - push back to queue and continue the loop.
                queuePush(run_queue_info->run_queue, rinfo);
                continue;
            }

            if (currentOpBatchable) {
                bool batchReady = true;
                batch_rinfo =
                    _BGThread_BatchOperations(run_queue_info, rinfo, batch_rinfo, &batchReady);
                if (!batchReady) {
                    // Batch is not ready - batch size didn't match the expectations from
                    // minbatchsize
                    for (int i = array_len(batch_rinfo) - 1; i >= 0; i--) {
                        queuePush(run_queue_info->run_queue, batch_rinfo[i]);
                    }
                    // Exit the loop, give a chance to new tasks to submit.
                    break;
                }
            }
            // Run the computation step (batched or not)
            // We're done with the queue here, items have been evicted so we can
            // safely unlock the queue mutex, to allow other threads to operate
            // on the same queue. The evicted items at this point are only visible
            // to this worker.
            pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
            _BGThread_Execute(run_queue_info, batch_rinfo);
            // Lock the queue again: we're done operating on evicted items only.
            pthread_mutex_lock(&run_queue_info->run_queue_mutex);
            _BGThread_ExecutionFinish(run_queue_info, batch_rinfo);
        }
    }
    array_free(batch_rinfo);
}
