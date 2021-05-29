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

#pragma once

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <pthread.h>

#include "config/config.h"
#include "DAG/dag.h"
#include "redis_ai_objects/model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "redis_ai_objects/script.h"
#include "redis_ai_objects/stats.h"
#include "redis_ai_objects/tensor.h"
#include "util/arr.h"
#include "util/rax.h"
#include "util/queue.h"

rax *RunQueues;
long long ThreadPoolSizePerQueue;

typedef struct RunQueueInfo {
    pthread_mutex_t run_queue_mutex;
    pthread_cond_t queue_condition_var;
    queue *run_queue;
    pthread_t *threads;
    pthread_key_t thread_id_key; // A key for getting the thread id from its local storage.
    char *device_str;
} RunQueueInfo;

typedef struct WorkerThreadInfo {
    RunQueueInfo *run_queue_info;
    int id;
} WorkerThreadInfo;

void RunQueueInfoFree(RunQueueInfo *info);

RunQueueInfo *CreateRunQueue(const char *device_str);

bool IsRunQueueExists(const char *device_str);

pthread_key_t GetQueueThreadIdKey(const char *device_str);

long long GetNumThreadsPerQueue(void);
