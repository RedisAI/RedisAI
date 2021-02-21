#define REDISMODULE_MAIN

typedef unsigned int uint;

#include "err.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "onnx_redisai.h"
#include "backends/onnxruntime.h"
#include "config.h"
#include "err.h"
#include <limits.h>

size_t _get_memory_usage(RedisModuleCtx *ctx) {
    RedisModuleCallReply *reply;
    reply = RedisModule_Call(ctx, "INFO", "c", "memory");
    if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING) {
        // Retrieve the used_memory field sub string and convert it to a number
        size_t len;
        const char *info_str = RedisModule_CallReplyStringPtr(reply, &len);
        const char *used_memory_str = strchr(info_str, ':') + 1;
        char *ptr;
        size_t used_memory = strtoul(used_memory_str, &ptr, 10);
        printf("used_memory: %zu\n", used_memory);
        RedisModule_FreeCallReply(reply);
        return used_memory;
    }
    return 0;
}

int RAI_onnxAllocator_modelSet(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if(argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    size_t memory_usage_before = _get_memory_usage(ctx);
    RAI_ModelOpts opts = {0};
    RAI_Error err = {0};
    size_t model_len = 26454;

    // Get onnx Model
    FILE *fp;
    char model[model_len];
    size_t index = 0;
    char model_path[1000];
    strcat(getcwd(model_path, sizeof(model_path)), "/../../RedisAI/tests/flow/test_data/linear_iris.onnx");
    fp = fopen(model_path, "r");
    char c;
    while((c = getc(fp)) != EOF) {
        model[index++] = c;
    }
    fclose(fp);
    model_len = index;
    RedisAI_LoadDefaultBackend(ctx, RAI_BACKEND_ONNXRUNTIME);
    RedisAI_ModelCreate(RAI_BACKEND_ONNXRUNTIME, "CPU", NULL, opts, 0, NULL, 0, NULL,
      model, model_len, &err);
    size_t memory_usage_after = _get_memory_usage(ctx);
    printf("Redis allocated : %zu\n", memory_usage_after-memory_usage_before);
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if(RedisModule_Init(ctx, "RAI_onnxAllocator", 1, REDISMODULE_APIVER_1) ==
       REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if(RedisAI_OnnxInitialize(ctx) != REDISMODULE_OK)
        RedisModule_Log(ctx, "warning",
          "could not initialize RedisAI api, running without AI support.");

    if(RedisModule_CreateCommand(ctx, "RAI_onnxAllocator.modelSet",
      RAI_onnxAllocator_modelSet, "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    return REDISMODULE_OK;
}