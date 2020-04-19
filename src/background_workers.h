#ifndef SRC_BACKGROUND_WORKERS_H_
#define SRC_BACKGROUND_WORKERS_H_

#include <pthread.h>

#include "config.h"
#include "dag.h"
#include "model.h"
#include "model_script_run_session.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "script.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

AI_dict *run_queues;
long long perqueueThreadPoolSize;


typedef struct RunQueueInfo {
  pthread_mutex_t run_queue_mutex;
  pthread_cond_t queue_condition_var;
  queue *run_queue;
  pthread_t *threads;
} RunQueueInfo;

int freeRunQueueInfo(RunQueueInfo *info);

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char *devicestr, RunQueueInfo **run_queue_info);

#endif /* SRC_BACKGROUND_WORKERS_H_ */
