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
#include "util/dict.h"
#include "util/queue.h"

AI_dict *run_queues;
long long perqueueThreadPoolSize;

typedef struct RunQueueInfo {
    pthread_mutex_t run_queue_mutex;
    pthread_cond_t queue_condition_var;
    queue *run_queue;
    pthread_t *threads;
    char *devicestr;
} RunQueueInfo;

int freeRunQueueInfo(RunQueueInfo *info);

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char *devicestr, RunQueueInfo **run_queue_info);
