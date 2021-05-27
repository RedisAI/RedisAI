#include "onnx_timeout.h"
#include "util/arr.h"
#include <sys/time.h>

// Gets the current time in milliseconds.
static long long _mstime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust/1000;
}

int CreateGlobalOnnxRunSessions(long long size) {
    OnnxRunSessions = array_new(OnnxRunSessionCtx *, size);
    for (size_t i = 0; i < size; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Calloc(1, sizeof(OnnxRunSessionCtx));
        OnnxRunSessions = array_append(OnnxRunSessions, entry);
    }
    return REDISMODULE_OK;
}

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid,
  uint64_t subevent, void *data) {

    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    size_t len = array_len(OnnxRunSessions);
    for (size_t i = 0; i < len; i++) {
        if (OnnxRunSessions[i]->runOptions == NULL) {
            continue;
        }
        long long curr_time = _mstime();
        if (curr_time - OnnxRunSessions[i]->queuingTime > ONNX_MAX_RUNTIME) {
            ort->RunOptionsSetTerminate(OnnxRunSessions[i]->runOptions);
        }
    }
}

void SetRunSessionCtx(size_t index, OrtRunOptions *newRunOptions) {
    OnnxRunSessionCtx *runSessionCtx = OnnxRunSessions[index];
    RedisModule_Assert(runSessionCtx->runOptions == NULL);
    runSessionCtx->runOptions = newRunOptions;
    runSessionCtx->queuingTime = _mstime();
}

void ClearRunSessionCtx(size_t index) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    OnnxRunSessionCtx *runSessionCtx = OnnxRunSessions[index];
    ort->ReleaseRunOptions(runSessionCtx->runOptions);
    runSessionCtx->runOptions = NULL;
}
