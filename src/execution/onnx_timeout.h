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

// This is a global rax that holds an array of OnnxRunSessionCtx for every device
// that onnx models may run on.
rax *OnnxRunSessions;

int CreateGlobalOnnxRunSessions(void);

int AddDeviceToGlobalRunSessions(const char *device);

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                                void *data);

OnnxRunSessionCtx *SetGetRunSessionCtx(const char *device, OrtRunOptions *new_run_options);

void ClearRunSessionCtx(OnnxRunSessionCtx *run_session_ctx);
