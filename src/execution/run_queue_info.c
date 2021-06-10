#include "string_utils.h"
#include "run_queue_info.h"
#include "backends/backends.h"
#include "background_workers.h"

RunQueueInfo *RunQueue_Create(const char *device_str) {

    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    RAI_StringToUpper(device_str, upper_device_str, device_str_len + 1);

    // Create new run queue and initialize its inner fields.
    RunQueueInfo *run_queue_info = RedisModule_Alloc(sizeof(RunQueueInfo));
    run_queue_info->run_queue = queueCreate();
    run_queue_info->device_str = RedisModule_Strdup(upper_device_str);
    pthread_cond_init(&(run_queue_info->queue_condition_var), NULL);
    pthread_mutex_init(&(run_queue_info->run_queue_mutex), NULL);
    run_queue_info->threads =
        (pthread_t *)RedisModule_Alloc(sizeof(pthread_t) * Config_GetNumThreadsPerQueue());

    // Save device with its associate run queue info in the dictionary.
    if (AI_dictAdd(RunQueues, upper_device_str, run_queue_info) != DICT_OK) {
        RunQueue_Free(run_queue_info);
        return NULL;
    }

    // Create worker threads.
    for (int i = 0; i < Config_GetNumThreadsPerQueue(); i++) {
        if (pthread_create(&(run_queue_info->threads[i]), NULL, BGWorker_ThreadMain,
                           run_queue_info) != 0) {
            AI_dictDelete(RunQueues, upper_device_str);
            RunQueue_Free(run_queue_info);
            return NULL;
        }
    }

    // Add the new device worker threads to onnx run sessions tracking.
    if (RAI_backends.onnx.add_new_device) {
        RAI_backends.onnx.add_new_device(device_str);
    }
    return run_queue_info;
}

RunQueueInfo *RunQueue_GetInfo(const char *device_str) {
    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    RAI_StringToUpper(device_str, upper_device_str, device_str_len + 1);
    AI_dictEntry *entry = AI_dictFind(RunQueues, upper_device_str);
    RedisModule_Assert(entry != NULL);
    return AI_dictGetVal(entry);
}

bool RunQueue_IsExists(const char *device_str) {
    size_t device_str_len = strlen(device_str);
    char upper_device_str[device_str_len + 1];
    RAI_StringToUpper(device_str, upper_device_str, device_str_len + 1);
    if (AI_dictFind(RunQueues, upper_device_str) == NULL) {
        return false;
    }
    return true;
}

void RunQueue_Free(RunQueueInfo *run_queue_info) {
    RedisModule_Assert(queueLength(run_queue_info->run_queue) == 0);
    RedisModule_Free(run_queue_info->run_queue);
    RedisModule_Free(run_queue_info->device_str);

    // Wait for workers to exit and free the pool.
    for (int i = 0; i < Config_GetNumThreadsPerQueue(); i++) {
        RedisModule_Assert(pthread_join(run_queue_info->threads[i], NULL) == 0);
        RedisModule_Free(run_queue_info->threads);
    }
    pthread_mutex_destroy(&(run_queue_info->run_queue_mutex));
    pthread_cond_destroy(&(run_queue_info->queue_condition_var));
    RedisModule_Free(run_queue_info);
}
