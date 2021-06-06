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
#include "util/queue.h"

AI_dict *RunQueues;
long long ThreadPoolSizePerQueue; // Number of working threads for device.
uintptr_t BGWorkersCounter;       // Total number of BG threads running currently.
pthread_key_t ThreadIdKey;        // Holds the thread id in its local storage.

typedef struct RunQueueInfo {
    pthread_mutex_t run_queue_mutex;
    pthread_cond_t queue_condition_var;
    queue *run_queue;
    pthread_t *threads;
    char *device_str;
} RunQueueInfo;

/**
 * @brief Terminate all working threads and free the run queue with its inner fields.
 */
void RunQueueInfoFree(RunQueueInfo *info);

/**
 * @brief Create a new run queue for a device.
 */
RunQueueInfo *CreateRunQueue(const char *device_str);

/**
 * @brief Return true if a ru queue exists for this particular device.
 */
bool IsRunQueueExists(const char *device_str);

/**
 * @brief Return the RunQueueInfo saved in the global RunQueues dict for a certain
 * device name, or NULL if doesn't exist.
 */
RunQueueInfo *GetRunQueueInfo(const char *device_str);

/**
 * @brief Return the thread id from its local storage by accessing the value
 * saved under ThreadIdKey.
 */
uintptr_t GetThreadId(void);
