#pragma once

#include "backends/onnxruntime.h"
#include "onnxruntime_c_api.h"

// The maximum time in milliseconds before killing onnx run session.
#define ONNX_MAX_RUNTIME 5000

typedef struct onnxRunSessionCtx {
    long long queuingTime;
    OrtRunOptions* runOptions;
} onnxRunSessionCtx;

onnxRunSessionCtx **OnnxRunSessions;

int CreateGlobalOnnxRunSessions(pthread_t *working_thread_ids, size_t size)

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid,
  uint64_t subevent, void *data);

void ReplaceRunSessionCtx(size_t index, OrtRunOptions *runOptions);
