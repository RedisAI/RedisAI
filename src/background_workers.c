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
#include "model_script_run_session.h"
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
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <ctype.h>

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

char* strToUpper(const char* input) {
  char* output = RedisModule_Strdup(input);
  size_t output_len = strlen(output);
  for (long long i=0; i<output_len; i++) {
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
    (*run_queue_info)->threads = (pthread_t *)RedisModule_Alloc(
        sizeof(pthread_t) * perqueueThreadPoolSize);
    /* create threads */
    for (int i = 0; i < perqueueThreadPoolSize; i++) {
      if (pthread_create(&((*run_queue_info)->threads[i]), NULL,
                         RedisAI_Run_ThreadMain, *run_queue_info) != 0) {
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
  while (true) {
    int rc = pthread_cond_wait(&run_queue_info->queue_condition_var,
                               &run_queue_info->run_queue_mutex);

    // This is the length of the queue for this particular device
    // (see run_queue_info->devicestr).
    // There might be more than one thread operating on the same
    // queue, according to the THREADS_PER_QUEUE config variable.
    long long run_queue_len = queueLength(run_queue_info->run_queue);

    while (run_queue_len > 0) {
      queueItem **evicted_items = NULL;
      RedisAI_RunInfo **batch_rinfo = NULL;

      // We first evict the front of the queue
      queueItem *item = queueFront(run_queue_info->run_queue);

      // TODO DAG BATCHING
      // If a batch item is a DAG and relies on unrealized outputs, just skip it
      // and give way to other items, since the client is surely different (given
      // the fact that commands are blocking for the individual clent, so that
      // the temporal sequence of commands cannot be broken).

      while (item) {
        RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)item->value;

        // We keep a list of additional items that are evicted from the queue
        // to find oppotunities for batching
        if (evicted_items) {
          array_free(evicted_items);
          array_free(batch_rinfo);
        }
        evicted_items = array_new(queueItem *, run_queue_len);
        batch_rinfo = array_new(RedisAI_RunInfo *, run_queue_len);

        // We add the current item to the list of evicted items. If it's the
        // first time around this will be the queue front.
        evicted_items = array_append(evicted_items, item);
        batch_rinfo = array_append(batch_rinfo, rinfo);

        // TODO DAG REFACTORING: we need to move the batching logic out of here 
        // and have it at the DAG logic level.
        // We need to know what's next on the DAG. Remember that here we are
        // already on a specific device.
        // Look into RAI_DagRunSessionStep.

        // If it's a DAG (signaled by the fact that use_local_context equals 1)
        // then stop looking for matching items to batch because batching with
        // DAG commands is not yet supported.
        // TODO DAG BATCHING
        // if (rinfo->use_local_context==1){
        //   break;
        // }

        int batchable_op;
        size_t batchsize, minbatchsize;
        RedisAI_DagCurrentOpBatchingInfo(rinfo, run_queue_info->devicestr,
                                         &batchable_op,
                                         &batchsize, &minbatchsize);

        // If the op is not batchable (e.g. scriptrun), we stop looking for matching
        // items to batch
        if (batchable_op == 0) {
          break;
        }

        // If no batching was requested on the first item, then skip the search
        if (batchsize == 0) {
          break;
        }

        // Get the size of the batch so far, that is, the size of the first input
        // tensor in the 0-th dimension
        // TODO DAG REFACTORING: rinfo should be dagop
        size_t current_batchsize = RAI_RunInfoBatchSize(rinfo);

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
          // TODO DAG BATCHING
          // make sure we lock properly when we access DagOps
          // TODO DAG REFACTORING: next_rinfo should be dagop instead
          if (RAI_RunInfoBatchable(rinfo, next_rinfo) == 0) {
            next_item = queueNext(next_item);
            continue;
          }

          // Get size of batch of the selected matching item in the queue
          // TODO DAG REFACTORING: next_rinfo should be dagop instead
          int next_batchsize = RAI_RunInfoBatchSize(next_rinfo);

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


        // If minbatchsize wasn't been set, or if the current batch
        // size exceeds the minimum batch size already, then we're done
        // Otherwise, if minbatchsize was set and the size wasn't reached,
        // loop until there's something new on the queue
        if (minbatchsize == 0 || current_batchsize >= minbatchsize) {
          break;
        }

        item = item->next;
      }

      // If there was nothing on the queue, free up the arrays and unlock
      // so we can wait for more items to land on the queue
      if (item == NULL) {
        array_free(evicted_items);
        array_free(batch_rinfo);
        pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
        break;
      }

      // We're ready to process the items in the evicted list, so actually
      // evict them from the queue
      for (long long i = 0; i < array_len(evicted_items); i++) {
        queueEvict(run_queue_info->run_queue, evicted_items[i]);
      }

      // We're done with the queue here, items have been evicted so we can
      // safely unlock the queue mutex, to allow other threads to operate
      // on the same queue. The evicted items at this point are only visible
      // to this worker.
      pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

      // Run the computation step. Here we have two cases:
      // 1. the first run info in the batch (since DAG is not batchable) is a DAG
      // 2. no dag in the batch
      int dag_progress = 0;
      int dag_device_complete = 0;
      int dag_all_devices_complete = 0;
      if (array_len(batch_rinfo) > 0) {
        // Run the first unrealized DAG operation. Several variables can be
        // filled after the function returns:
        // dag_progress: there has been progress in the DAG, that is, the first
        //               unrealized step has produced a result; if not, it means
        //               that the inputs of the first unrealized step were not
        //               available yet
        // dag_device_complete: all DAG operations for the current device have completed,
        //                      there's nothing more this worker can do
        // dag_all_devices_complete: all DAG operations for any device have completed,
        //                           which means that the computation of the DAG is complete
        RedisAI_DagRunSessionStep(batch_rinfo[0], run_queue_info->devicestr,
                                  &dag_progress, &dag_device_complete, &dag_all_devices_complete);
 
        // if (batch_rinfo[0]->use_local_context == 1) {
        //   // Run the first unrealized DAG operation. Several variables can be
        //   // filled after the function returns:
        //   // dag_progress: there has been progress in the DAG, that is, the first
        //   //               unrealized step has produced a result; if not, it means
        //   //               that the inputs of the first unrealized step were not
        //   //               available yet
        //   // dag_device_complete: all DAG operations for the current device have completed,
        //   //                      there's nothing more this worker can do
        //   // dag_all_devices_complete: all DAG operations for any device have completed,
        //   //                           which means that the computation of the DAG is complete
        //   RedisAI_DagRunSessionStep(batch_rinfo[0], run_queue_info->devicestr,
        //                             &dag_progress, &dag_device_complete, &dag_all_devices_complete);
        // } else {
        //   // Run the (potentially batched) model or the (non-batched) script run
        //   RAI_ModelRunScriptRunSession(batch_rinfo);
        // }
      }

      // We don't need the run info container for the batch anymore at this point
      array_free(batch_rinfo);

      // Lock the queue again: we're done operating on evicted items only, we need
      // to update the queue with the new information after run
      pthread_mutex_lock(&run_queue_info->run_queue_mutex);

      // Loop over evicted items
      for (long long i = 0; i < array_len(evicted_items); i++) {
        // Get the current evicted run info
        RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[i]->value);
        // const int use_local_context = evicted_rinfo->use_local_context;
        const int single_op_dag = evicted_rinfo->single_op_dag;
        const int single_device_dag = evicted_rinfo->single_device_dag;
        int dagError = 0;
        int dagRefCount = -1;
        // If it's a multi-op DAG
        // TODO DAG REFACTORING: we could manage single_device_dag similarly (without locks)
        // if (use_local_context == 1) {
        if (single_op_dag == 0) {
          // Lock the DAG: a DAG might have more workers operating on different queues
          // operating on the same DAG operations and on a few shared variables, such as:
          // dagError: was there any error any DAG operation so far
          // dagRefCount: what is the count of active computations right now; the client
          //              can be eventually unblocked if and only if the ref count is zero
          //              otherwise a worker currently running a session might find itself
          //              without the DAG structure, since unblocking leads to structures
          //              being released
          // TODO DAG REFACTORING: we need to be more conservative with locking, by relying
          // on information about the topology of the DAG (single op, single device, ...)
          pthread_mutex_lock(evicted_rinfo->dagMutex);
          dagError = *evicted_rinfo->dagError;
          dagRefCount = *evicted_rinfo->dagRefCount;
          pthread_mutex_unlock(evicted_rinfo->dagMutex);

          // If there's still more to do on the device queue and there was no error
          if (dag_device_complete == 0 && !dagError) {
            // If there was progress in the DAG, aka the current step produced a result
            if (dag_progress) {
              // Put the run info back at the front of the queue, so that this worker
              // will be able to proceed with the computation
              queuePushFront(run_queue_info->run_queue, evicted_rinfo);
            }
            // Otherwise, if there was no progress on the DAG, meaning the inputs to the
            // current operations were not ready (they depend on other workers on other
            // queues completing their job)
            else {
              // If the queue is not empty, ie if the DAG is not the only item in the
              // queue
              if (queueLength(run_queue_info->run_queue) > 0) {
                // Pop the next item in the queue
                queueItem *next_item = queuePop(run_queue_info->run_queue);
                RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;
                // Push the DAG to the front of the queue, and then the item we just
                // popped in front of it, so that it becomes the first item in the queue.
                // The rational is, since the DAG needs to wait for other workers, we are
                // giving way to the next item and we'll get back to the DAG when that is done
                queuePushFront(run_queue_info->run_queue, evicted_rinfo);
                queuePushFront(run_queue_info->run_queue, next_rinfo);
              }
              // If there's nothing else in the queue
              else {
                // We push the DAG back at the front
                queuePushFront(run_queue_info->run_queue, evicted_rinfo);
                // Sleep for a millisecond since there's nothing else on the queue that has a
                // chance to run, we give other workers a chance to produce the inputs needed
                // for this DAG step
                usleep(1000);
              }
            }
          }

          // If DAG operations have completed on all queues, or if any DAG operation on any
          // queue produced an error, and if there's no worker with running computations, and
          // if the client is still connected
          if ((dag_all_devices_complete || dagError) && dagRefCount == 0 && evicted_rinfo->client) {
            // Unblock the client
            RedisModule_UnblockClient(evicted_rinfo->client, evicted_rinfo);
          }

          // If there's nothing else to do for the DAG in the current worker or if an error
          // occurred in any worker, we just move on
          if (dag_device_complete || dagError) {
            RedisModule_Free(evicted_items[i]);
          }
        }
        // If we're done processing a modelrun or scriptrun command
        else {
          // If the client is still connected
          if (evicted_rinfo->client) {
            // Unblock the client. Note that we do this for every evicted info in a batch,
            // which will be associated with a different client. Here we unblock every
            // client appropriately
            RedisModule_UnblockClient(evicted_rinfo->client, evicted_rinfo);
          }
          RedisModule_Free(evicted_items[i]);
        }
      }
      array_free(evicted_items);

      run_queue_len = queueLength(run_queue_info->run_queue);
    }
  }
}
