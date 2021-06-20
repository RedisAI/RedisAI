#pragma once

#include <stdint.h>

/**
 * @return The internal id of RedisAI current working thread.
 * id range is {0, ..., <threads_count>-1}. If this is called from a non
 * RedisAI BG thread, return -1.
 */
long (*RedisAI_GetThreadId)(void);

/**
 * @return The number of working threads in RedisAI. This number should be
 * equal to the number of threads per queue (load time config) * number of devices
 * registered in RedisAI (a new device is registered if a model is set to run on
 * this device in AI.MODELSTORE command.
 */
uintptr_t (*RedisAI_GetThreadsCount)(void);

/**
 * @return The number of working threads per device queue (load time config).
 */
long long (*RedisAI_GetNumThreadsPerQueue)(void);

/**
 * @return The maximal number of milliseconds that a model run session should run
 * before it is terminated forcefully (load time config).
 * Currently supported only fo onnxruntime backend.
 */
long long (*RedisAI_GetModelExecutionTimeout)(void);
