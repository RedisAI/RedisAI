#pragma once
#include "redisai.h"
#include <pthread.h>
#include <stdbool.h>

#define LLAPIMODULE_OK 0
#define LLAPIMODULE_ERR 1

typedef struct RAI_RunResults {
    RAI_Tensor **outputs;
    RAI_Error *error;
} RAI_RunResults;

int testLoadTensor(RedisModuleCtx *ctx);

int testModelRunOpError(RedisModuleCtx *ctx);

int testEmptyDAGError(RedisModuleCtx *ctx);

int testKeysMismatchError(RedisModuleCtx *ctx);

int testBuildDAGFromString(RedisModuleCtx *ctx);

int testSimpleDAGRun(RedisModuleCtx *ctx);

int testSimpleDAGRun2(RedisModuleCtx *ctx);

int testSimpleDAGRun2Error(RedisModuleCtx *ctx);

int testDAGResnet(RedisModuleCtx *ctx);
