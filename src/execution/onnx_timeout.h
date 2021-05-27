#pragma once

#include "backends/onnxruntime.h"
#include "onnxruntime_c_api.h"

// The maximum time in milliseconds before killing onnx run session.
#define ONNX_MAX_RUNTIME 5000

typedef struct OnnxRunSessionCtx {
    long long queuingTime;
    OrtRunOptions* runOptions;
} OnnxRunSessionCtx;

OnnxRunSessionCtx **OnnxRunSessions;

int CreateGlobalOnnxRunSessions(long long size);

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid,
  uint64_t subevent, void *data);

void SetRunSessionCtx(size_t index, OrtRunOptions *newRunOptions);

void ClearRunSessionCtx(size_t index);
