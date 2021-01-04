#pragma once

#include "redisai.h"

RAI_DAGRunCtx *RedisAI_DagRunCtxCreate(void);

int RAI_DagAddModelRun_(RAI_DAGRunCtx *run_info, RAI_ModelRunCtx *mctx, RedisModuleString **inputs,
                        RedisModuleString **outputs, RAI_Error *err);

int RAI_DagAddModelRun(RAI_DAGRunCtx *run_info, RAI_ModelRunCtx *mctx, const char **inputs,
                       size_t ninputs, const char **outputs, size_t noutputs, RAI_Error *err);

int RAI_DagAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err);