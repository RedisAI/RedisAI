/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "redisai.h"
#include <pthread.h>
#include <stdbool.h>

#define LLAPIMODULE_OK  0
#define LLAPIMODULE_ERR 1

typedef struct RAI_RunCtx {
    RAI_Tensor **outputs;
    RAI_Error *error;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} RAI_RunCtx;

int testLoadTensor(RedisModuleCtx *ctx);

int testModelRunOpError(RedisModuleCtx *ctx);

int testEmptyDAGError(RedisModuleCtx *ctx);

int testKeysMismatchError(RedisModuleCtx *ctx);

int testBuildDAGFromString(RedisModuleCtx *ctx);

int testSimpleDAGRun(RedisModuleCtx *ctx);

int testSimpleDAGRun2(RedisModuleCtx *ctx);

int testSimpleDAGRun2Error(RedisModuleCtx *ctx);

int testDAGResnet(RedisModuleCtx *ctx);
