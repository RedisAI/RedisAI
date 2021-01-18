#pragma once
#include "../../src/redisai.h"
#include <pthread.h>

#define LLAPIMODULE_OK 0
#define LLAPIMODULE_ERR 1

typedef struct RAI_RunResults {
    RAI_Tensor **outputs;
    RAI_Error *error;
} RAI_RunResults;

int testModelRunOpError(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info);

int testEmptyDAGError(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info);

int testKeysMismatchError(RedisModuleCtx *ctx,RAI_DAGRunCtx *run_info);

int testBuildDAGFromString(RedisModuleCtx *ctx,RAI_DAGRunCtx *run_info);

int testSimpleDAGRun(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info);

int testSimpleDAGRun2(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info);

int testSimpleDAGRun2Error(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info);

int testDAGResnet(RedisModuleCtx *ctx);
