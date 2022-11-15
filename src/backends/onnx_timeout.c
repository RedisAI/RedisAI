#include "onnx_timeout.h"
#include "util/arr.h"
/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "execution/utils.h"
#include "config/config.h"
#include <pthread.h>
#include "util/string_utils.h"
#include "redis_ai_objects/stats.h"
#include "backends_api.h"

int RAI_InitGlobalRunSessionsORT() {
    onnx_global_run_sessions = RedisModule_Alloc(sizeof(OnnxGlobalRunSessions));

    // Initialize the array with entries number equals to the number of currently
    // working threads in RedisAI (note that CPU threads must exist form the start).
    size_t RAI_working_threads_num = RedisAI_GetThreadsCount();
    OnnxRunSessionCtx **run_sessions_array =
        array_new(OnnxRunSessionCtx *, RAI_working_threads_num);
    for (size_t i = 0; i < RAI_working_threads_num; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Alloc(sizeof(OnnxRunSessionCtx));
        entry->runState = RedisModule_Alloc(sizeof(entry->runState));
        *entry->runState = RUN_SESSION_AVAILABLE;
        entry->queuingTime = LLONG_MAX;
        run_sessions_array = array_append(run_sessions_array, entry);
    }
    onnx_global_run_sessions->OnnxRunSessions = run_sessions_array;
    pthread_rwlock_init(&(onnx_global_run_sessions->rwlock), NULL);

    return REDISMODULE_OK;
}

size_t RAI_GetGlobalRunSessionsLenORT() {
    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    size_t len = array_len(onnx_global_run_sessions->OnnxRunSessions);
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
    return len;
}

int RAI_AddNewDeviceORT(const char *device_str) {

    // Acquire write lock, as we might reallocate the array while extending it.
    pthread_rwlock_wrlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx **run_sessions_array = onnx_global_run_sessions->OnnxRunSessions;

    // Extend the array with an entry for every working thread on the new device,
    // initialized to NULL.
    size_t size = RedisAI_GetNumThreadsPerQueue();
    for (size_t i = 0; i < size; i++) {
        OnnxRunSessionCtx *entry = RedisModule_Alloc(sizeof(OnnxRunSessionCtx));
        entry->runState = RedisModule_Alloc(sizeof(entry->runState));
        *entry->runState = RUN_SESSION_AVAILABLE;
        entry->queuingTime = LLONG_MAX;
        run_sessions_array = array_append(run_sessions_array, entry);
    }
    onnx_global_run_sessions->OnnxRunSessions = run_sessions_array;
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
    return REDISMODULE_OK;
}

void RAI_EnforceTimeoutORT(RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent,
                           void *data) {
    RedisModule_Assert(eid.id == REDISMODULE_EVENT_CRON_LOOP);
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx **run_sessions_ctx = onnx_global_run_sessions->OnnxRunSessions;
    size_t len = array_len(run_sessions_ctx);
    long long curr_time = mstime();
    long long timeout = RedisAI_GetModelExecutionTimeout();
    for (size_t i = 0; i < len; i++) {
        // Check if a sessions is running for too long, and kill it if is still active.
        // If entry doesn't contain active session, its queueing time is LLONG_MAX
        // (thus the following condition will always be evaluated as false)
        if (curr_time - __atomic_load_n(&(run_sessions_ctx[i]->queuingTime), __ATOMIC_RELAXED) >
            timeout) {
            if (__sync_bool_compare_and_swap(run_sessions_ctx[i]->runState, RUN_SESSION_ACTIVE,
                                             RUN_SESSION_INVALID)) {
                // Set termination flag, validate that ONNX API succeeded (returns NULL)
                RedisModule_Assert(ort->RunOptionsSetTerminate(run_sessions_ctx[i]->runOptions) ==
                                   NULL);
                __atomic_store_n(run_sessions_ctx[i]->runState, RUN_SESSION_TERMINATED,
                                 __ATOMIC_RELAXED);
            }
        }
    }
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}

void RAI_ActivateRunSessionCtxORT(OrtRunOptions *new_run_options, long *run_session_index) {

    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    // Get the thread id (which is the correspondent index in the global sessions array + 1).
    // if thread id is -1, we are not running from RedisAI thread (not allowed)
    *run_session_index = RedisAI_GetThreadId();
    if (*run_session_index == -1) {
        pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
        return;
    }
    OnnxRunSessionCtx *entry = onnx_global_run_sessions->OnnxRunSessions[*run_session_index];
    RedisModule_Assert(*entry->runState == RUN_SESSION_AVAILABLE);
    RedisModule_Assert(entry->queuingTime == LLONG_MAX);

    // Update the entry with the current session data.
    entry->runOptions = new_run_options;
    __atomic_store_n(&(entry->queuingTime), mstime(), __ATOMIC_RELAXED);
    __atomic_store_n(entry->runState, RUN_SESSION_ACTIVE, __ATOMIC_RELAXED);
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}

void RAI_ResetRunSessionCtxORT(long run_session_index) {
    const OrtApi *ort = OrtGetApiBase()->GetApi(1);
    pthread_rwlock_rdlock(&(onnx_global_run_sessions->rwlock));
    OnnxRunSessionCtx *entry = onnx_global_run_sessions->OnnxRunSessions[run_session_index];

    // In most cases, state will be ACTIVE at this point, and we want to turn in to
    // AVAILABLE atomically, so we won't call the kill switch at the same time.
    if (!__sync_bool_compare_and_swap(entry->runState, RUN_SESSION_ACTIVE, RUN_SESSION_AVAILABLE)) {
        // If state was not ACTIVE, it is INVALID/TERMINATED, due to a timeout that
        // has occurred. We do busy wait until the state is set to TERMINATE.
        while (!__sync_bool_compare_and_swap(entry->runState, RUN_SESSION_TERMINATED,
                                             RUN_SESSION_AVAILABLE))
            ;
    }
    __atomic_store_n(&(entry->queuingTime), LLONG_MAX, __ATOMIC_RELAXED);
    ort->ReleaseRunOptions(entry->runOptions);
    pthread_rwlock_unlock(&(onnx_global_run_sessions->rwlock));
}
