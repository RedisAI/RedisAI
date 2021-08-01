#pragma once

/**
 * Contains the structure to manage the per-device queues, used for decoupling
 * the work from the main thread to the background worker threads. For each of
 * the incoming ModelRun, ScriptRun, and DagRun commands, the request is queued
 * and evaded asynchronously to one the device queues.
 */

#include "utils.h"
#include "queue.h"
#include "dictionaries.h"

extern AI_dict *RunQueues;

typedef struct RunQueueInfo {
    pthread_mutex_t run_queue_mutex;
    pthread_cond_t queue_condition_var;
    queue *run_queue;
    pthread_t *threads;
    char *device_str;
} RunQueueInfo;

/**
 * @brief Create a new run queue for a device.
 */
RunQueueInfo *RunQueue_Create(const char *device_str);

/**
 * @brief Return true if a ru queue exists for this particular device.
 */
bool RunQueue_IsExists(const char *device_str);

/**
 * @brief Return the RunQueueInfo saved in the global RunQueues dict for a certain
 * device name, after asserting that it exists.
 */
RunQueueInfo *RunQueue_GetInfo(const char *device_str);

/**
 * @brief Terminate all working threads and free the run queue with its inner fields.
 */
void RunQueue_Free(RunQueueInfo *info);
