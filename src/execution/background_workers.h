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

/**
 * @brief RedisAI main loop for every background working thread
 * @param arg - This is the run queue info of the device on which this thread is
 * running the AI model/script
 */
void *BGWorker_ThreadMain(void *arg);

/**
 * @brief Returns the thread id (among RedisAI working threads). If this is called
 * form a non RedisAI working thread, return -1
 */
long BGWorker_GetThreadId(void);

/**
 * @brief Returns the total number of RedisAI working threads (for all devices).
 */
uintptr_t BGWorker_GetThreadsCount(void);