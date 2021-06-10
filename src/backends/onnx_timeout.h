#pragma once

#include "backends/onnxruntime.h"
#include "onnxruntime_c_api.h"

typedef struct OnnxRunSessionCtx {
    long long queuingTime;
    OrtRunOptions *runOptions;
} OnnxRunSessionCtx;

// This is a global array of OnnxRunSessionCtx. Contains an entry for every thread
// (on every device) that onnx models may run on.
typedef struct OnnxGlobalRunSessions {
    OnnxRunSessionCtx **OnnxRunSessions;
    pthread_rwlock_t rwlock;
} OnnxGlobalRunSessions;

OnnxGlobalRunSessions *onnx_global_run_sessions;

/**
 * @brief This is called whenever Onnx backend is loaded. It creates the global
 * OnnxGlobalRunSessions structure with entry-per-thread (for CPU threads at first),
 * so that every thread will have a designated entry to update with the onnx session
 * that it's going to run.
 */
int RAI_InitGlobalRunSessionsORT(void);

/**
 * @return The length of the global array (should be the number of current working threads)
 */
size_t RAI_GetGlobalRunSessionsLenORT(void);

/**
 * @brief This is called whenever RedisAI gets a request to store a model that run
 * on a new device, and creates some more working thread, as configured in
 * ThreadPerQueue. Thus, the global array of onnx sessions that has an
 * entry-per-thread is extended accordingly.
 */
int RAI_AddNewDeviceORT(const char *device_str);

/**
 * @brief A callback that is registered to RedisCron event, that is, it is called
 * periodically and go over all the (possibly running) onnx sessions, and kill
 * those that exceeds the timeout.
 */
void RAI_EnforceTimeoutORT(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                           void *data);

/**
 * @brief Set a new OrtRunOptions in the global structure, to allow us to
 * "terminate" the run session from the cron callback.
 * @param new_run_options - The newly created OrtRunOptions to store.
 * @param run_session_index - placeholder for the index of the running thread
 * in the global array, to have a quick access later to clean this entry.
 */
void RAI_SetRunSessionCtxORT(OrtRunOptions *new_run_options, size_t *run_session_index);

/**
 * @brief Release the OrtRunOptions of a session that finished its run and
 * reset the corresponding entry in the global structure.
 * @param run_session_index - The entry index where OrtRunOptions was stored.
 */
void RAI_InvalidateRunSessionCtxORT(size_t run_session_index);
