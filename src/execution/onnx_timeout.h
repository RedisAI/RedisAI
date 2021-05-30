#pragma once

#include "backends/onnxruntime.h"
#include "onnxruntime_c_api.h"
#include "util/rax.h"

// The maximum time in milliseconds before killing onnx run session.
// todo: make it a load time config
#define ONNX_MAX_RUNTIME 5000

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
int CreateGlobalOnnxRunSessions(void);

/**
 * @brief This is called whenever RedisAI gets a request to store a model that run
 * on a new device, and creates some more working thread, as configured in
 * ThreadPerQueue. Thus, the global array of onnx sessions that has an
 * entry-per-thread is extended accordingly.
 */
int ExtendGlobalRunSessions(const char *device_str);

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                                void *data);

void SetRunSessionCtx(OrtRunOptions *new_run_options, size_t *run_session_index);

void ClearRunSessionCtx(size_t run_session_index);
