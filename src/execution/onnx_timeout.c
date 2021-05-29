#include "onnx_timeout.h"
#include "util/arr.h"
#include <sys/time.h>
#include <pthread.h>
#include "util/rax.h"
#include "util/string_utils.h"

// Gets the current time in milliseconds.
static long long _mstime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust / 1000;
}

int CreateGlobalOnnxRunSessions() {
    OnnxRunSessions = raxNew();
    if (OnnxRunSessions == NULL) {
        return REDISMODULE_ERR;
    }
    return AddDeviceToGlobalRunSessions("CPU");
}

int AddDeviceToGlobalRunSessions(const char *device) {

    size_t size = RedisAI_NumThreadsPerQueue();
    // Create array with an entry for every working thread, initialized to NULL.
    OnnxRunSessionCtx **device_run_sessions = array_new(OnnxRunSessionCtx *, size);
    for (size_t i = 0; i < size; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Calloc(1, sizeof(OnnxRunSessionCtx));
        device_run_sessions = array_append(device_run_sessions, entry);
    }
    // Add the array to the global rax that holds onnx run sessions per device.
    size_t device_str_len = strlen(device);
    char upper_device_str[device_str_len + 1];
    String_ToUpper(device, upper_device_str, &device_str_len);
    if (raxInsert(OnnxRunSessions, (unsigned char *)upper_device_str, device_str_len,
                  device_run_sessions, NULL) != 1) {
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

void OnnxEnforceTimeoutCallback(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                                void *data) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);

    raxIterator rax_it;
    raxStart(&rax_it, OnnxRunSessions);
    raxSeek(&rax_it, "^", NULL, 0);

    // Go over all the possible existing run sessions for every device.
    while (raxNext(&rax_it)) {
        OnnxRunSessionCtx **onnx_run_sessions_per_device = rax_it.data;
        size_t threads_per_device = array_len(onnx_run_sessions_per_device);
        for (size_t i = 0; i < threads_per_device; i++) {
            if (onnx_run_sessions_per_device[i]->runOptions == NULL) {
                continue;
            }
            long long curr_time = _mstime();
            // Check if a sessions is running for too long, and kill it if so.
            if (curr_time - onnx_run_sessions_per_device[i]->queuingTime > ONNX_MAX_RUNTIME) {
                ort->RunOptionsSetTerminate(onnx_run_sessions_per_device[i]->runOptions);
            }
        }
    }
}

OnnxRunSessionCtx *SetGetRunSessionCtx(const char *device, OrtRunOptions *new_run_options) {

    int *thread_ind = (int *)pthread_getspecific(RedisAI_ThreadIdKey());

    OnnxRunSessionCtx **device_run_sessions =
        raxFind(OnnxRunSessions, (unsigned char *)device, strlen(device));
    RedisModule_Assert(device_run_sessions != raxNotFound);
    RedisModule_Assert(device_run_sessions[*thread_ind]->runOptions == NULL);

    device_run_sessions[*thread_ind]->runOptions = new_run_options;
    device_run_sessions[*thread_ind]->queuingTime = _mstime();
    return device_run_sessions[*thread_ind];
}

void ClearRunSessionCtx(OnnxRunSessionCtx *run_session_ctx) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    ort->ReleaseRunOptions(run_session_ctx->runOptions);
    run_session_ctx->runOptions = NULL;
}
