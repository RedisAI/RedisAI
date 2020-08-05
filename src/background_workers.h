/**
 * background_workers.h
 *
 * Contains the structure and method signatures required to manage the
 * per-device queues, used for decoupling the work from the main thread to the
 * background worker threads. For each of the incoming ModelRun, ScriptRun, and
 * DagRun commands, the request is queued and evaded asynchronously to one the
 * device queues.
 *
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

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

AI_dict *run_queues;
long long perqueueThreadPoolSize;

typedef struct RunQueueInfo {
  pthread_mutex_t run_queue_mutex;
  pthread_cond_t queue_condition_var;
  queue *run_queue;
  pthread_t *threads;
  char* devicestr;
} RunQueueInfo;

int freeRunQueueInfo(RunQueueInfo *info);

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char *devicestr, RunQueueInfo **run_queue_info);

#endif /* SRC_BACKGROUND_WORKERS_H_ */
