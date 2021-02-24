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
#include "path.h"

typedef struct MemoryInfo {
    size_t usage;
    size_t access_counter;
} MemoryInfo;

MemoryInfo _get_memory_info() {

    RedisModuleServerInfoData *server_info = RedisModule_GetServerInfo(NULL, "everything");
    int onnx_info_error;
    unsigned long long onnx_counter = RedisModule_ServerInfoGetFieldUnsigned(server_info,
      "ai_onnxruntime_memory_access_num", &onnx_info_error);
    RedisModule_Assert(onnx_info_error == 0);
    unsigned long long onnx_mem = RedisModule_ServerInfoGetFieldUnsigned(server_info,
      "ai_onnxruntime_memory", &onnx_info_error);
    RedisModule_Assert(onnx_info_error == 0);
    RedisModule_FreeServerInfo(NULL, server_info);
    return (MemoryInfo){.usage = onnx_mem, .access_counter = onnx_counter};
}

int RAI_onnxAllocator_modelSet(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if(argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    RedisAI_LoadDefaultBackend(ctx, RAI_BACKEND_ONNXRUNTIME);
    RedisModule_Assert(_get_memory_info().usage == 0);

    // Load onnx model to local buffer.
    size_t model_len = 130;
    FILE *fp;
    char model_data[model_len];
    size_t index = 0;
    char model_path[1024];
    strcpy(model_path, path);
    strcat(model_path, "/test_data/mul_1.onnx");
    fp = fopen(model_path, "rb");
    char c;
    while((c = fgetc(fp)) != EOF) {
        model_data[index++] = c;
    }
    fclose(fp);
    RedisModule_Assert(model_len == index);

    // Create model and verify that onnx backend uses REDIS allocator.
    RAI_ModelOpts opts = {0};
    RAI_Error err = {0};
    RAI_Model *model = RedisAI_ModelCreate(RAI_BACKEND_ONNXRUNTIME, "CPU", NULL, opts, 0, NULL, 0, NULL,
      model_data, model_len, &err);
    RedisModule_Assert(err.code == RAI_OK);

    // Expect 3 allocation access by onnx backend for allocating model, input name
    // and output name
    MemoryInfo mem_info = _get_memory_info();
    if (mem_info.access_counter != 3 || mem_info.usage == 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "model set allocation error");
    }

    RedisAI_ModelFree(model, &err);
    RedisModule_Assert(err.code == RAI_OK);
    mem_info = _get_memory_info();
    if (mem_info.access_counter != 6 || mem_info.usage != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "model free error");
    }
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