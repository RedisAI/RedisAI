#include "onnx_timeout.h"
#include "util/arr.h"
#include <sys/time.h>

// Gets the current time in milliseconds.
long long _mstime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust/1000;
}

int CreateGlobalOnnxRunSessions(pthread_t *working_thread_ids, size_t size) {
    OnnxRunSessions = array_new(onnxRunSessionCtx *, size);
    for (size_t i = 0; i < size; i++) {
        OnnxRunSessions = array_append(OnnxRunSessions, NULL);
    }
    return REDISMODULE_OK;
}

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid,
  uint64_t subevent, void *data) {

    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    size_t len = array_len(OnnxRunSessions);
    for (size_t i = 0; i < len; i++) {
        if (OnnxRunSessions[i] == NULL) {
            continue;
        }
        long long currTime = _mstime();
        if (currTime - OnnxRunSessions[i]->queuingTime > ONNX_MAX_RUNTIME) {
            ort->RunOptionsSetTerminate(OnnxRunSessions[i]->runOptions);
        }
    }
}

void ReplaceRunSessionCtx(size_t index, OrtRunOptions *newRunOptions) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    if (OnnxRunSessions[index] != NULL) {
        ort->ReleaseRunOptions(OnnxRunSessions[index]->runOptions);
        RedisModule_Free(OnnxRunSessions[index]);
    }
    onnxRunSessionCtx *runSessionCtx = RedisModule_Alloc(sizeof(onnxRunSessionCtx));
    runSessionCtx->runOptions = newRunOptions;
    runSessionCtx->queuingTime = _mstime();
}
