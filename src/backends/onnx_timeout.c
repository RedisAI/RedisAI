#include "onnx_timeout.h"
#include "util/arr.h"
#include "util.h"
#include "execution/utils.h"
#include "config/config.h"
#include <pthread.h>
#include "util/string_utils.h"
#include "redis_ai_objects/stats.h"


int CreateGlobalOnnxRunSessions() {
    onnx_global_run_sessions = RedisModule_Alloc(sizeof(struct OnnxGlobalRunSessions));

    // Initialize the array with entries number equals to the number of currently
    // working threads in RedisAI (note that CPU threads must exist form the start).
    size_t RAI_working_threads_num = RedisAI_GetThreadsCount();
    OnnxRunSessionCtx **run_sessions_array =
        array_new(OnnxRunSessionCtx *, RAI_working_threads_num);
    for (size_t i = 0; i < RAI_working_threads_num; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Calloc(1, sizeof(OnnxRunSessionCtx));
        run_sessions_array = array_append(run_sessions_array, entry);
    }
    onnx_global_run_sessions->OnnxRunSessions = run_sessions_array;
    pthread_rwlock_init(&(onnx_global_run_sessions->rwlock), NULL);

    return REDISMODULE_OK;
}

int RAI_AddNewDeviceORT(const char *device_str) {

    // Acquire write lock, as we might reallocate the array while extending it.
    pthread_rwlock_wrlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx **run_sessions_array = onnx_global_run_sessions->OnnxRunSessions;

    // Extend the array with an entry for every working thread on the new device,
    // initialized to NULL.
    size_t size = RedisAI_GetNumThreadsPerQueue();
    for (size_t i = 0; i < size; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Calloc(1, sizeof(OnnxRunSessionCtx));
        run_sessions_array = array_append(run_sessions_array, entry);
    }
    onnx_global_run_sessions->OnnxRunSessions = run_sessions_array;
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
    return REDISMODULE_OK;
}

void RAI_EnforceTimeoutORT(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                           void *data) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx **run_sessions_ctx = onnx_global_run_sessions->OnnxRunSessions;
    size_t len = array_len(run_sessions_ctx);
    for (size_t i = 0; i < len; i++) {
        if (run_sessions_ctx[i]->runOptions == NULL) {
            continue;
        }
        long long curr_time = mstime();
        long long timeout = RedisAI_GetModelExecutionTimeout();
        // Check if a sessions is running for too long, and kill it if so.
        if (curr_time - run_sessions_ctx[i]->queuingTime > timeout) {
            ort->RunOptionsSetTerminate(run_sessions_ctx[i]->runOptions);
        }
    }
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}

void SetRunSessionCtx(OrtRunOptions *new_run_options, size_t *run_session_index) {

    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    // Get the thread index (which is the correspondent index in the global sessions array).
    *run_session_index = RedisAI_GetThreadId();
    OnnxRunSessionCtx *entry = onnx_global_run_sessions->OnnxRunSessions[*run_session_index];
    RedisModule_Assert(entry->runOptions == NULL);

    // Update the entry with the current session data.
    entry->runOptions = new_run_options;
    entry->queuingTime = mstime();
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}

void InvalidateRunSessionCtx(size_t run_session_index) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx *entry = onnx_global_run_sessions->OnnxRunSessions[run_session_index];
    ort->ReleaseRunOptions(entry->runOptions);
    entry->runOptions = NULL;
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}
