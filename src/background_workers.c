/**
 * background_workers.c
 *
 * Contains the structure to manage the per-device queues, used for decoupling
 * the work from the main thread to the background worker threads. For each of
 * the incoming ModelRun, ScriptRun, and DagRun commands, the request is queued
 * and evaded asynchronously to one the device queues.
 *
 */

#include "background_workers.h"
#include "dag.h"
#include "model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "script.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int freeRunQueueInfo(RunQueueInfo *info) {
    int result = REDISMODULE_OK;
    if (info->run_queue) {
        RedisModule_Free(info->run_queue);
    }
    RedisModule_Free(info->devicestr);
    if (info->threads) {
        /* Wait for workers to exit */
        for (int i = 0; i < perqueueThreadPoolSize; i++) {
            const int rtn = pthread_join(info->threads[i], NULL);
            if (rtn != 0) {
                result = REDISMODULE_ERR;
            }
        }
        /* Now free pool structure */
        RedisModule_Free(info->threads);
    }
    RedisModule_Free(info);
    return result;
}

void *RedisAI_Run_ThreadMain(void *arg);

char *strToUpper(const char *input) {
    char *output = RedisModule_Strdup(input);
    size_t output_len = strlen(output);
    for (long long i = 0; i < output_len; i++) {
        output[i] = toupper(output[i]);
    }
    return output;
}

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char *devicestr, RunQueueInfo **run_queue_info) {
    int result = REDISMODULE_ERR;
    if (run_queues == NULL) {
        return result;
    }

    char *devicestr_ = strToUpper(devicestr);

    AI_dictEntry *entry = AI_dictFind(run_queues, devicestr_);
    if (entry) {
        *run_queue_info = AI_dictGetVal(entry);
        result = REDISMODULE_OK;
    } else {
        *run_queue_info = RedisModule_Alloc(sizeof(RunQueueInfo));
        (*run_queue_info)->run_queue = queueCreate();
        (*run_queue_info)->devicestr = RedisModule_Strdup(devicestr_);
        pthread_cond_init(&(*run_queue_info)->queue_condition_var, NULL);
        pthread_mutex_init(&(*run_queue_info)->run_queue_mutex, NULL);
        (*run_queue_info)->threads =
            (pthread_t *)RedisModule_Alloc(sizeof(pthread_t) * perqueueThreadPoolSize);
        /* create threads */
        for (int i = 0; i < perqueueThreadPoolSize; i++) {
            if (pthread_create(&((*run_queue_info)->threads[i]), NULL, RedisAI_Run_ThreadMain,
                               *run_queue_info) != 0) {
                freeRunQueueInfo(*run_queue_info);
                return REDISMODULE_ERR;
            }
        }
        AI_dictAdd(run_queues, (void *)devicestr_, (void *)*run_queue_info);
        result = REDISMODULE_OK;
    }

    RedisModule_Free(devicestr_);

    return result;
}

void *RedisAI_Run_ThreadMain(void *arg) {
    RunQueueInfo *run_queue_info = (RunQueueInfo *)arg;
    pthread_t self = pthread_self();
    RAI_PTHREAD_SETNAME("redisai_bthread");
    pthread_mutex_lock(&run_queue_info->run_queue_mutex);

    queueItem **evicted_items = array_new(queueItem *, 1);
    RedisAI_RunInfo **batch_rinfo = array_new(RedisAI_RunInfo *, 1);

    while (true) {

        // In case a DAG Op can express a MINBATCHSIZE > 0 with a MINBATCHTIMEOUT
        // in milliseconds, we will use a timedwait of one millisecond to evaluate
        // whether the run needs to trigger in case nothing else happens on the
        // queue.
        // Possible optimization: opt for a longer timeout if there's no
        // minbatchtimeout involved.
        struct timeval now;
        gettimeofday(&now, NULL);

        struct timespec absTimeout;
        absTimeout.tv_sec = now.tv_sec;
        absTimeout.tv_nsec = (now.tv_usec + 1000) * 1000; // 1 millisecond

        int rc = pthread_cond_timedwait(&run_queue_info->queue_condition_var,
                                        &run_queue_info->run_queue_mutex, &absTimeout);

        // This is the length of the queue for this particular device
        // (see run_queue_info->devicestr).
        // There might be more than one thread operating on the same
        // queue, according to the THREADS_PER_QUEUE config variable.
        long long run_queue_len = queueLength(run_queue_info->run_queue);

        while (run_queue_len > 0) {
            // We first peek the front of the queue
            queueItem *item = queueFront(run_queue_info->run_queue);

            // We define a few variables to inform about the next action to take
            // based on the status of the entries on the queue
            // Do we need to unblock the client (or clients in case of batching)?
            int do_unblock = 0;
            // Do we need to run the entry (or the entries in case of batching)?
            int do_run = 0;
            // Is the entry not ready to run due to inputs not being available yet,
            // so that we need to put the entry back on the queue?
            int do_retry = 0;
            // Were all ops in entry executed for the device, so that we can just
            // remove the entry from the queue and be done with it?
            int device_complete = 0;

            while (item) {
                RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)item->value;

                // We keep a list of additional items that are evicted from the queue
                // to find opporunities for batching
                array_clear(evicted_items);
                array_clear(batch_rinfo);

                if (rinfo->timeout > 0) {
                    int timedOut = __atomic_load_n(rinfo->timedOut, __ATOMIC_RELAXED);

                    if (timedOut == 0) {
                        struct timeval now, sub;
                        gettimeofday(&now, NULL);
                        timersub(&now, &rinfo->queuingTime, &sub);
                        size_t time_msec = sub.tv_sec * 1000 + sub.tv_usec / 1000;

                        if (time_msec > rinfo->timeout) {
                            timedOut = 1;
                            __atomic_store_n(rinfo->timedOut, timedOut, __ATOMIC_RELAXED);
                        }
                    }

                    if (timedOut == 1) {
                        queueEvict(run_queue_info->run_queue, item);

                        long long dagRefCount = RAI_DagRunInfoFreeShallowCopy(rinfo);
                        if (dagRefCount == 0 && rinfo->client) {
                            RedisModule_UnblockClient(rinfo->client, rinfo);
                        }

                        queueItem *evicted_item = item;
                        item = item->next;
                        RedisModule_Free(evicted_item);

                        continue;
                    }
                }

                // Since we might be looking through the queue for candidates, we need
                // to reinitialize our findings every time we consider a new item for
                // running
                do_unblock = 0;
                do_run = 0;
                do_retry = 0;
                device_complete = 0;

                // We add the current item to the list of evicted items. If it's the
                // first time around this will be the queue front.
                evicted_items = array_append(evicted_items, item);
                batch_rinfo = array_append(batch_rinfo, rinfo);

                // Get the currentOp from the DAG at the top of the queue
                // The currentOp is the first op without a result that needs to run
                // on the device for the queue
                // Also collect info on:
                // - whether the currentOp is ready (all its inputs
                //   are present in the context)
                // - whether the currentOp is batchable (that is, it is a model and
                //   it was set with batchsize > 0)
                RAI_DagOp *currentOp = RedisAI_DagCurrentOp(rinfo);

                // Have all ops that need to run on the device been executed
                int deviceComplete = RedisAI_DagDeviceComplete(rinfo);
                // Have all ops in the DAG been executed
                int dagComplete = RedisAI_DagComplete(rinfo);

                // If all ops in the DAG, then we decide we will unblock (one thread
                // will actually unblock in practice, according to the reference count)
                if (dagComplete) {
                    device_complete = 1;
                    do_unblock = 1;
                    break;
                }

                // If we made it to here, we won't unblock (unless there's an error
                // during the run, see below)
                do_unblock = 0;

                // If all ops on the device have a result, then we don't schedule to run
                // and we won't place the entry back on the queue
                if (deviceComplete) {
                    do_run = 0;
                    do_retry = 0;
                    device_complete = 1;
                    break;
                }

                // Is the currentOp ready to run (all its inputs are present in the
                // context) Is the currentOp batchable (that is, it is a modelrun and it
                // was set with batchsize > 0)
                int currentOpReady, currentOpBatchable;
                RedisAI_DagCurrentOpInfo(rinfo, &currentOpReady, &currentOpBatchable);

                // If any of the inputs of the current op is not in the context, it
                // means that some parent ops did not execute. In this case we don't
                // schedule to run, but we will place the entry back on the queue
                if (currentOpReady == 0) {
                    do_run = 0;
                    do_retry = 1;
                    break;
                }

                // If we made it this far, we will run the currentOp
                do_run = 1;

                // If the current op is not batchable (that is, if it's not a modelrun
                // or if it's a modelrun but batchsize was set to 0), we stop looking
                // further
                if (currentOpBatchable == 0) {
                    break;
                }

                // If we are here, then we scheduled to run and we currently have an
                // operation that can be batched.

                // Since the current op can be batched, then we collect info on
                // batching, namely
                // - batchsize
                // - minbatchsize
                // - minbatchtimeout
                // - actual size of the input along the batch (0-th) dimension
                size_t batchsize, minbatchsize, minbatchtimeout, inbatchsize;
                RedisAI_DagOpBatchInfo(rinfo, currentOp, &batchsize, &minbatchsize,
                                       &minbatchtimeout, &inbatchsize);

                // Get the size of the batch so far, that is, the size of the first
                // input tensor in the 0-th dimension
                size_t current_batchsize = inbatchsize;

                // If the size is zero or if it already exceeds the desired batch size
                // then stop searching
                if (current_batchsize == 0 || current_batchsize >= batchsize) {
                    break;
                }

                // Get the next item in the queue
                queueItem *next_item = item->next;

                // While we don't reach the end of the queue
                while (next_item != NULL) {
                    // Get the next run info
                    RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;

                    // If the next item is batchable, that is, if it is a model, if it
                    // is a call to the same model, and if the size of the inputs except
                    // the 0-th dimension match, then go on, otherwise continue to the
                    // next item in the queue
                    RAI_DagOp *nextOp = RedisAI_DagCurrentOp(next_rinfo);

                    int nextOpReady, nextOpBatchable;
                    RedisAI_DagCurrentOpInfo(next_rinfo, &nextOpReady, &nextOpBatchable);

                    if (nextOpReady == 0 || nextOpBatchable == 0) {
                        next_item = queueNext(next_item);
                        continue;
                    }

                    int batched = 0;
                    size_t next_batchsize = 0;
                    RedisAI_DagOpBatchingMatch(rinfo, currentOp, next_rinfo, nextOp, &batched,
                                               &next_batchsize);

                    if (batched == 0) {
                        next_item = queueNext(next_item);
                        continue;
                    }

                    // If the new batch size would exceed the prescribed batch
                    // size, then quit searching.
                    // Here we could consider searching further down the queue.
                    if (current_batchsize + next_batchsize > batchsize) {
                        break;
                    }

                    // If all previous checks pass, then keep track of the item
                    // in the list of evicted items
                    evicted_items = array_append(evicted_items, next_item);
                    batch_rinfo = array_append(batch_rinfo, next_rinfo);

                    // Update the batchsize and go to the next item to see if
                    // there's anything else to batch
                    current_batchsize += next_batchsize;
                    next_item = queueNext(next_item);
                }

                // If minbatchsize hasn't been set, or if the current batch
                // size exceeds the minimum batch size already, then we're done.
                // Otherwise, if minbatchsize was set and the size wasn't reached,
                // loop until there's something new on the queue
                if (minbatchsize == 0 || current_batchsize >= minbatchsize) {
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
                        break;
                    }
                }

                item = item->next;
            }

            // If there was nothing on the queue, free up the arrays and unlock
            // so we can wait for more items to land on the queue
            if (item == NULL) {
                break;
            }

            // We're ready to process the items in the evicted list, so actually
            // evict them from the queue
            for (long long i = 0; i < array_len(evicted_items); i++) {
                queueEvict(run_queue_info->run_queue, evicted_items[i]);
            }

            // For better readability, we create a variable that says if the current
            // run will actually be batched or not
            int batched_run = array_len(batch_rinfo) > 1;

            // Variable holding the fact that running triggered an error.
            int run_error = 0;
            // Run the computation step (batched or not)
            if (do_run == 1) {
                // We're done with the queue here, items have been evicted so we can
                // safely unlock the queue mutex, to allow other threads to operate
                // on the same queue. The evicted items at this point are only visible
                // to this worker.
                pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

                // For simplicity, we call into different functions whether the run
                // is batched or not
                if (batched_run == 1) {
                    RedisAI_BatchedDagRunSessionStep(batch_rinfo, run_queue_info->devicestr);
                } else {
                    RedisAI_DagRunSessionStep(batch_rinfo[0], run_queue_info->devicestr);
                }

                // Lock the queue again: we're done operating on evicted items only, we
                // need to update the queue with the new information after run
                pthread_mutex_lock(&run_queue_info->run_queue_mutex);

                // Run is over, now iterate over the run info structs in the batch
                // and see if any error was generated
                int dagError = 0;
                for (long long i = 0; i < array_len(batch_rinfo); i++) {
                    RedisAI_RunInfo *rinfo = batch_rinfo[i];
                    // We lock on the DAG because error could be set from
                    // other threads operating on the same DAG (TODO: use atomic)
                    dagError = __atomic_load_n(rinfo->dagError, __ATOMIC_RELAXED);

                    // We record that there was an error for later on
                    run_error = dagError;

                    // If there was an error and the reference count for the dag
                    // has gone to zero and the client is still around, we unblock
                    if (dagError) {
                        long long dagRefCount = RAI_DagRunInfoFreeShallowCopy(rinfo);
                        if (dagRefCount == 0 && rinfo->client) {
                            RedisModule_UnblockClient(rinfo->client, rinfo);
                        }
                    } else {
                        rinfo->dagDeviceCompleteOpCount += 1;
                        __atomic_add_fetch(rinfo->dagCompleteOpCount, 1, __ATOMIC_RELAXED);
                    }
                }
            }

            // We initialize variables where we'll store the fact hat, after the
            // current run, all ops for the device or all ops in the dag could be
            // complete. This way we can avoid placing the op back on the queue if
            // there's nothing left to do.
            int device_complete_after_run = RedisAI_DagDeviceComplete(batch_rinfo[0]);
            int dag_complete_after_run = RedisAI_DagComplete(batch_rinfo[0]);

            long long dagRefCount = -1;

            if (device_complete == 1 || device_complete_after_run == 1) {
                RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[0]->value);
                // We decrease and get the reference count for the DAG
                dagRefCount = RAI_DagRunInfoFreeShallowCopy(evicted_rinfo);
            }

            // If the DAG was complete, then it's time to unblock the client
            if (do_unblock == 1 || dag_complete_after_run == 1) {
                RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[0]->value);

                // If the reference count for the DAG is zero and the client is still
                // around, then we actually unblock the client
                if (dagRefCount == 0 && evicted_rinfo->client) {
                    RedisModule_UnblockClient(evicted_rinfo->client, evicted_rinfo);
                }
            }

            // If there was no progress on the DAG, meaning the inputs to the
            // current operation were not ready (they depend on other workers on other
            // queues completing their job), then we put the entry back on the queue
            if (do_retry == 1) {
                RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[0]->value);

                if (queueLength(run_queue_info->run_queue) > 0) {
                    // Pop the next item in the queue
                    queueItem *next_item = queuePop(run_queue_info->run_queue);
                    RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;
                    // Push the DAG to the front of the queue, and then the item we just
                    // popped in front of it, so that it becomes the first item in the
                    // queue. The rationale is, since the DAG needs to wait for other
                    // workers, we are giving way to the next item and we'll get back to
                    // the DAG when that is done
                    queuePushFront(run_queue_info->run_queue, evicted_rinfo);
                    queuePushFront(run_queue_info->run_queue, next_rinfo);
                }
                // If there's nothing else in the queue
                else {
                    // We push the DAG back at the front
                    queuePushFront(run_queue_info->run_queue, evicted_rinfo);
                    // Since there's nothing else on the queue we just break out and give
                    // other workers a chance to produce the inputs needed for this DAG
                    // step
                    break;
                }
            }

            // If the op was ran successfully and without any error, then put the
            // entry back on the queue unless all ops for the device have been
            // executed
            if (do_run == 1 && run_error == 0) {
                // Here we iterate backwards to keep the first evicted on top
                // A side effect of this is that we are potentially changing priority in
                // the queue We could solve this using a priority queue, TODO for later
                for (long long i = array_len(evicted_items) - 1; i >= 0; i--) {
                    // Get the current evicted run info
                    RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[i]->value);
                    // If the DAG for the top-most item is complete for the device, then
                    // we don't push it back on the queue
                    if (i == 0 && device_complete_after_run == 1) {
                        continue;
                    }
                    queuePushFront(run_queue_info->run_queue, evicted_rinfo);
                }
            }

            // TODO now we can figure out of the device is complete or the dag is
            // complete if (dag_complete_op_count == evicted_rinfo[0]->dagOpCount) ->
            // ublock, free if (dag_device_complete_op_count ==
            // evicted_rinfo[0]->dagDeviceOpCount) -> device complete

            // If there's nothing else to do for the DAG in the current worker or if
            // an error occurred in any worker, we just move on
            if (device_complete == 1 || device_complete_after_run == 1 || do_unblock == 1 ||
                run_error == 1) {
                for (long long i = 0; i < array_len(evicted_items); i++) {
                    RedisModule_Free(evicted_items[i]);
                }
            }

            run_queue_len = queueLength(run_queue_info->run_queue);
        }
    }

    array_free(evicted_items);
    array_free(batch_rinfo);
}
