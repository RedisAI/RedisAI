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

int freeRunQueueInfo(RunQueueInfo *info) {
  int result = REDISMODULE_OK;
  if (info->run_queue) {
    RedisModule_Free(info->run_queue);
  }
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

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char *devicestr, RunQueueInfo **run_queue_info) {
  int result = REDISMODULE_ERR;
  if (run_queues == NULL) {
    return result;
  }

  AI_dictEntry *entry = AI_dictFind(run_queues, devicestr);
  if (entry) {
    *run_queue_info = AI_dictGetVal(entry);
    result = REDISMODULE_OK;
  } else {
    *run_queue_info = RedisModule_Alloc(sizeof(RunQueueInfo));
    (*run_queue_info)->run_queue = queueCreate();
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
    AI_dictAdd(run_queues, (void *)devicestr, (void *)*run_queue_info);
    result = REDISMODULE_OK;
  }

  return result;
}

void *RedisAI_Run_ThreadMain(void *arg) {
  RunQueueInfo *run_queue_info = (RunQueueInfo *)arg;

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  while (true) {
    int rc = pthread_cond_wait(&run_queue_info->queue_condition_var,
                               &run_queue_info->run_queue_mutex);

    long long run_queue_len = queueLength(run_queue_info->run_queue);

    while (run_queue_len > 0) {
      queueItem **evicted_items = NULL;
      RedisAI_RunInfo **batch_rinfo = NULL;

      queueItem *item = queueFront(run_queue_info->run_queue);

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

        // DAGRUN
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

      if (array_len(batch_rinfo) > 0) {
        if (batch_rinfo[0]->use_local_context == 1) {
          RedisAI_DagRunSession(batch_rinfo[0]);
        } else {
          RAI_ModelRunScriptRunSession(batch_rinfo);
        }
      }

      for (long long i = 0; i < array_len(evicted_items); i++) {
        RedisModule_Free(evicted_items[i]);
      }
      array_free(evicted_items);
      array_free(batch_rinfo);

      pthread_mutex_lock(&run_queue_info->run_queue_mutex);

      run_queue_len = queueLength(run_queue_info->run_queue);
    }
  }
}