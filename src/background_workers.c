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
  for (long long i=0; i<strlen(output); i++) {
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

    long long run_queue_len = queueLength(run_queue_info->run_queue);

    while (run_queue_len > 0) {
      queueItem **evicted_items = NULL;
      RedisAI_RunInfo **batch_rinfo = NULL;

      queueItem *item = queueFront(run_queue_info->run_queue);

      // TODO DAG BATCHING
      // If a batch item is a DAG and relies on unrealized outputs, just skip it
      // and give way to other items, since the client is surely different (given
      // the fact that commands are blocking for the individual clent, so that
      // the temporal sequence of commands cannot be broken).

      while (item) {
        RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)item->value;

        if (evicted_items) {
          array_free(evicted_items);
          array_free(batch_rinfo);
        }
        evicted_items = array_new(queueItem *, run_queue_len);
        batch_rinfo = array_new(RedisAI_RunInfo *, run_queue_len);

        array_append(evicted_items, item);
        array_append(batch_rinfo, rinfo);

        if (rinfo->sctx) {
          break;
        }

        // TODO DAG BATCHING
        // With DAG refactoring we will be able to batch, we just
        // need to handle outputs properly.
        if (rinfo->use_local_context==1){
          break;
        }

        size_t batchsize = rinfo->mctx->model->opts.batchsize;

        if (batchsize == 0) {
          break;
        }

        size_t current_batchsize = RAI_RunInfoBatchSize(rinfo);

        if (current_batchsize == 0 || current_batchsize >= batchsize) {
          break;
        }

        queueItem *next_item = item->next;

        while (next_item != NULL) {
          RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;

          // TODO DAG BATCHING
          // make sure we lock properly when we access DagOps
          if (RAI_RunInfoBatchable(rinfo, next_rinfo) == 0) {
            next_item = queueNext(next_item);
            continue;
          }

          int next_batchsize = RAI_RunInfoBatchSize(next_rinfo);

          if (current_batchsize + next_batchsize > batchsize) {
            break;
          }

          array_append(evicted_items, next_item);
          array_append(batch_rinfo, next_rinfo);

          current_batchsize += next_batchsize;
          next_item = queueNext(next_item);
        }

        size_t minbatchsize = rinfo->mctx->model->opts.minbatchsize;

        if (minbatchsize == 0 || current_batchsize >= minbatchsize) {
          break;
        }

        item = item->next;
      }

      if (item == NULL) {
        array_free(evicted_items);
        array_free(batch_rinfo);
        pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
        break;
      }

      for (long long i = 0; i < array_len(evicted_items); i++) {
        queueEvict(run_queue_info->run_queue, evicted_items[i]);
      }

      pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

      int dag_progress = 0;
      int dag_complete = 0;
      if (array_len(batch_rinfo) > 0) {
        if (batch_rinfo[0]->use_local_context == 1) {
          RedisAI_DagRunSessionStep(batch_rinfo[0], run_queue_info->devicestr, &dag_progress, &dag_complete);
        } else {
          RAI_ModelRunScriptRunSession(batch_rinfo);
        }
      }

      array_free(batch_rinfo);

      pthread_mutex_lock(&run_queue_info->run_queue_mutex);

      for (long long i = 0; i < array_len(evicted_items); i++) {
        RedisAI_RunInfo *evicted_rinfo = (RedisAI_RunInfo *)(evicted_items[i]->value);
        const int use_local_context = evicted_rinfo->use_local_context;
        pthread_mutex_lock(evicted_rinfo->dagMutex);
        const int dagError = *evicted_rinfo->dagError;
        pthread_mutex_unlock(evicted_rinfo->dagMutex);
        if (use_local_context == 1 && dag_complete == 0 && !dagError) {
          if (dag_progress) {
            queueUnpop(run_queue_info->run_queue, evicted_rinfo);
          }
          else {
            if (queueLength(run_queue_info->run_queue) > 0) {
              queueItem *next_item = queuePop(run_queue_info->run_queue);
              RedisAI_RunInfo *next_rinfo = (RedisAI_RunInfo *)next_item->value;
              queueUnpop(run_queue_info->run_queue, evicted_rinfo);
              queueUnpop(run_queue_info->run_queue, next_rinfo);
            }
            else {
              queueUnpop(run_queue_info->run_queue, evicted_rinfo);
            }
          }
        }
        else {
          RedisModule_Free(evicted_items[i]);
        }
      }
      array_free(evicted_items);

      run_queue_len = queueLength(run_queue_info->run_queue);
    }
  }
}
