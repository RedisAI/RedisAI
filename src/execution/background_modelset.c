#include "background_modelset.h"
#include "command_parser.h"
#include "backends/backends.h"
#include <sys/time.h>
#include "backends/util.h"

#define BG_MODELSET_THREADS_NUM 1

int Init_BG_ModelSet() {

    modelSet_QueueInfo = RedisModule_Alloc(sizeof(RunQueueInfo));
    modelSet_QueueInfo->run_queue = queueCreate();
    modelSet_QueueInfo->devicestr = "";
    pthread_cond_init(&(modelSet_QueueInfo)->queue_condition_var, NULL);
    pthread_mutex_init(&(modelSet_QueueInfo)->run_queue_mutex, NULL);
    modelSet_QueueInfo->threads =
        (pthread_t *)RedisModule_Alloc(sizeof(pthread_t) * BG_MODELSET_THREADS_NUM);

    /* create thread(s) */
    for (int i = 0; i < BG_MODELSET_THREADS_NUM; i++) {
        if (pthread_create(&((modelSet_QueueInfo)->threads[i]), NULL, RedisAI_ModelSet_ThreadMain,
                           modelSet_QueueInfo) != 0) {
            freeRunQueueInfo(modelSet_QueueInfo);
            return REDISMODULE_ERR;
        }
    }
    return REDISMODULE_OK;
}

void ModelSet_Execute(void *args) {
    ModelSetCtx *model_ctx = (ModelSetCtx *)args;

    RedisModuleString **argv = model_ctx->args;
    model_ctx->model = RedisModule_Calloc(1, sizeof(*(model_ctx->model)));
    RAI_InitError(&model_ctx->err);
    RAI_Error *err = model_ctx->err;
    RAI_Model *model = model_ctx->model;
    model->refCount = 1;

    // If we fail, we unblock and the model_ctx internals will be freed.
    int status = ParseModelSetCommand(argv, array_len(argv), model, err);
    if (status != REDISMODULE_OK) {
        RedisModule_UnblockClient(model_ctx->client, model_ctx);
        return;
    }

    const char *backend_str = RAI_BackendName(model->backend);

    if (ModelCreateBE(model, err) != REDISMODULE_OK) {
        // If we got an error *not* because of lazy loading, we fail and unblock.
        if (RAI_GetErrorCode(err) != RAI_EBACKENDNOTLOADED) {
            RedisModule_UnblockClient(model_ctx->client, model_ctx);
            return;
        }
        RedisModule_Log(NULL, "warning", "backend %s not loaded, will try loading default backend",
                        backend_str);
        int ret = RAI_LoadDefaultBackend(NULL, model->backend);
        if (ret != REDISMODULE_OK) {
            RedisModule_Log(NULL, "error", "could not load %s default backend", backend_str);
            RedisModule_UnblockClient(model_ctx->client, model_ctx);
            return;
        }
        // Try creating model for backend again.
        RAI_ClearError(err);
        ModelCreateBE(model, err);
    }
    RedisModule_UnblockClient(model_ctx->client, model_ctx);
}

int RedisAI_ModelSet_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    ModelSetCtx *model_ctx = RedisModule_GetBlockedClientPrivateData(ctx);

    // If at some point we got an error, we return it (model_ctx is freed).
    if (RAI_GetErrorCode(model_ctx->err) != RAI_OK) {
        return RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(model_ctx->err));
    }

    // Save model in keyspace.
    RAI_Model *model = model_ctx->model;
    RedisModuleString *key_str = model->infokey;
    RedisModuleKey *key = RedisModule_OpenKey(ctx, key_str, REDISMODULE_READ | REDISMODULE_WRITE);
    int type = RedisModule_KeyType(key);

    // Two valid scenarios: 1. We create a new key, 2. The key is already holding
    // a RedisAI model type (in this case we update the key's value).
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RedisModule_ModuleTypeSetValue(key, RedisAI_ModelType, model);
    RedisModule_CloseKey(key);

    // Save this model in stats global dict.
    RAI_AddStatsEntry(NULL, key_str, RAI_MODEL, model->backend, model->devicestr, model->tag);

    // Get shallow copy so the free callback won't delete the model.
    RAI_ModelGetShallowCopy(model_ctx->model);

    RedisModule_ReplyWithSimpleString(ctx, "OK");

    // todo: this should be replaced with RedisModule_Replicate (currently not replicating properly)
    RedisModule_ReplicateVerbatim(ctx);
    return REDISMODULE_OK;
}

void ModelSet_FreeData(RedisModuleCtx *ctx, void *private_data) {
    ModelSetCtx *model_ctx = (ModelSetCtx *)private_data;

    RAI_FreeError(model_ctx->err);

    // This is a "dummy" error, we do not need it here since we only decrease
    // the model's ref_count in case of success, and otherwise a different error has returned.
    RAI_Error err = {0};
    if (model_ctx->model) {
        RAI_ModelFree(model_ctx->model, &err);
    }

    for (size_t i = 0; i < array_len(model_ctx->args); i++) {
        RedisModule_FreeString(NULL, model_ctx->args[i]);
    }
    array_free(model_ctx->args);

    RedisModule_Free(model_ctx);
}

void *RedisAI_ModelSet_ThreadMain(void *arg) {
    RunQueueInfo *run_queue_info = (RunQueueInfo *)arg;
    RAI_PTHREAD_SETNAME("redisai_modelset_bthread");
    pthread_mutex_lock(&run_queue_info->run_queue_mutex);

    while (true) {
        pthread_cond_wait(&run_queue_info->queue_condition_var, &run_queue_info->run_queue_mutex);
        queueItem *item = queuePop(run_queue_info->run_queue);
        pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

        // Currently the job's callback is always MODELSET.
        Job *job = queueItemGetValue(item);
        RedisModule_Free(item);
        job->Execute(job->args);
        JobFree(job);
        pthread_mutex_lock(&run_queue_info->run_queue_mutex);
    }
}

Job *JobCreate(void *args, void (*CallBack)(void *)) {
    Job *job = RedisModule_Alloc(sizeof(*job));
    job->args = args;
    job->Execute = CallBack;
}

void JobFree(Job *job) { RedisModule_Free(job); }